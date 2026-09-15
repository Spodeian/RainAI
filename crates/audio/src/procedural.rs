//! Procedural DDSP (Differentiable Digital Signal Processing) fallback synthesizer.
//!
//! Generates real-time 4-channel First Order Ambisonics (FOA) rain, wind, and side sounds
//! entirely in pure Rust with zero external allocations in the audio thread.

use crate::decoder::FoaFrame;
use shared::rain::{NoiseColor, RainState};

/// Fast, lightweight, deterministic pseudo-random number generator (Xorshift32)
#[derive(Clone, Debug)]
pub struct FastRng {
    state: u32,
}

impl FastRng {
    pub const fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x1234_5678 } else { seed },
        }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Generates float in [-1.0, 1.0]
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / 2_147_483_648.0) - 1.0
    }

    /// Generates float in [0.0, 1.0)
    #[inline]
    pub fn next_unit_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }
}

/// 3-pole Kellet filter for real-time Pink noise generation (-3 dB/oct)
#[derive(Clone, Debug, Default)]
pub struct PinkNoiseFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    b6: f32,
}

impl PinkNoiseFilter {
    #[inline]
    pub fn process(&mut self, white: f32) -> f32 {
        self.b0 = 0.99886 * self.b0 + white * 0.0555179;
        self.b1 = 0.99332 * self.b1 + white * 0.0750759;
        self.b2 = 0.96900 * self.b2 + white * 0.1538520;
        self.b3 = 0.86650 * self.b3 + white * 0.3104856;
        self.b4 = 0.55000 * self.b4 + white * 0.5329522;
        self.b5 = -0.7616 * self.b5 - white * 0.0168980;
        let pink = self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * 0.5362;
        self.b6 = white * 0.115926;
        pink * 0.11
    }
}

/// Single-pole lowpass filter for Brown noise (-6 dB/oct)
#[derive(Clone, Debug, Default)]
pub struct BrownNoiseFilter {
    last: f32,
}

impl BrownNoiseFilter {
    #[inline]
    pub fn process(&mut self, white: f32) -> f32 {
        self.last = (self.last * 0.96) + (white * 0.04);
        self.last * 3.5
    }
}

/// Simple 2-pole resonant biquad bandpass filter
#[derive(Clone, Debug)]
pub struct ResonantBandpass {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    b0: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Default for ResonantBandpass {
    fn default() -> Self {
        Self::new(1000.0, 2.0, 48000.0)
    }
}

impl ResonantBandpass {
    pub fn new(freq: f32, q: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            b0: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        filter.update(freq, q, sample_rate);
        filter
    }

    pub fn update(&mut self, freq: f32, q: f32, sample_rate: f32) {
        let omega = 2.0 * std::f32::consts::PI * (freq / sample_rate).clamp(0.001, 0.49);
        let alpha = omega.sin() / (2.0 * q.max(0.1));
        let cos_omega = omega.cos();

        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.b2 = -alpha / a0;
        self.a1 = (-2.0 * cos_omega) / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }
}

/// 16 learned drift factors from RainAI neural deployment config
pub const LEARNED_DRIFT: [f32; 16] = [
    0.0033373055, 0.0027257325, 0.0011902788, -0.0008779404,
    -0.0014967915, 0.0032215454, 0.0022632028, -0.0006406382,
    -0.0015108247, 0.0036037352, -0.0005228554, 0.0036691856,
    0.0023833837, 0.0039697173, 0.0030191161, 0.0034048420,
];

/// Nominal frequencies for the 16 subtractive DDSP bands (Hz)
pub const NOMINAL_BAND_FREQS: [f32; 16] = [
    2800.0, // 0: Corrugated Tin
    1400.0, // 1: Broad Leaves
    3200.0, // 2: Pine Needles
    900.0,  // 3: Pavement
    350.0,  // 4: Deep Water
    1800.0, // 5: Shallow Puddles
    480.0,  // 6: Canvas Tent
    4200.0, // 7: Glass Window
    650.0,  // 8: Wood Deck
    240.0,  // 9: Gutter / Downpipe Runoff
    1200.0, // 10: Bubble Chirps
    320.0,  // 11: Wind Howl
    6500.0, // 12: High Mist Spray
    180.0,  // 13: Low Air Turbulence
    2100.0, // 14: Foliage Rattle
    8500.0, // 15: Transducer Floor
];

/// Nominal Q resonance values for the 16 subtractive DDSP bands
pub const NOMINAL_BAND_Q: [f32; 16] = [
    4.5, // 0: Tin
    1.8, // 1: Broad Leaves
    3.8, // 2: Pine Needles
    1.2, // 3: Pavement
    1.5, // 4: Deep Water
    2.8, // 5: Shallow Puddles
    3.2, // 6: Canvas Tent
    4.0, // 7: Glass Window
    2.2, // 8: Wood Deck
    3.5, // 9: Downpipe Runoff
    6.0, // 10: Bubble Chirps
    6.0, // 11: Wind Howl
    1.5, // 12: High Mist Spray
    2.0, // 13: Low Air Turbulence
    2.5, // 14: Foliage Rattle
    1.0, // 15: Transducer Floor
];

/// 16-band subtractive parametric filterbank
#[derive(Clone, Debug)]
pub struct SubtractiveFilterbank16 {
    pub filters: [ResonantBandpass; 16],
}

impl SubtractiveFilterbank16 {
    pub fn new(sample_rate: f32) -> Self {
        let mut filters = [
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
            ResonantBandpass::default(),
        ];

        for i in 0..16 {
            let tuned_freq = NOMINAL_BAND_FREQS[i] * (1.0 + LEARNED_DRIFT[i]);
            filters[i] = ResonantBandpass::new(tuned_freq, NOMINAL_BAND_Q[i], sample_rate);
        }

        Self { filters }
    }
}

/// Real-time Procedural DDSP Synthesis Engine
#[derive(Clone, Debug)]
pub struct ProceduralSynthesizer {
    pub sample_rate: f32,
    rng: FastRng,
    pink_filter: PinkNoiseFilter,
    brown_filter: BrownNoiseFilter,
    filterbank: SubtractiveFilterbank16,
    thunder_rumble: f32,
    thunder_decay: f32,
    insect_phase: f32,
    bird_chirp_counter: usize,
    bird_chirp_duration: usize,
    bird_pitch: f32,
    traffic_filter: ResonantBandpass,
}

impl Default for ProceduralSynthesizer {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl ProceduralSynthesizer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            rng: FastRng::new(42),
            pink_filter: PinkNoiseFilter::default(),
            brown_filter: BrownNoiseFilter::default(),
            filterbank: SubtractiveFilterbank16::new(sample_rate),
            thunder_rumble: 0.0,
            thunder_decay: 0.9995,
            insect_phase: 0.0,
            bird_chirp_counter: 0,
            bird_chirp_duration: 1,
            bird_pitch: 3500.0,
            traffic_filter: ResonantBandpass::new(650.0, 1.8, sample_rate),
        }
    }

    /// Synthesizes one 4-channel FOA frame based on current RainState
    pub fn process_frame(&mut self, state: &RainState) -> FoaFrame {
        if !state.is_playing {
            return FoaFrame::default();
        }

        let white = self.rng.next_f32();

        // 1. Color-shaped base noise bed
        let base_noise = match state.noise_color {
            NoiseColor::White => white * 0.4,
            NoiseColor::Pink => self.pink_filter.process(white),
            NoiseColor::Brown => self.brown_filter.process(white),
            NoiseColor::Blue => (white - self.pink_filter.process(white)) * 0.5,
            NoiseColor::Violet => (white - self.brown_filter.process(white)) * 0.3,
        };

        // 2. Continuous 16-band subtractive parametric resonance response
        let rain_drive = base_noise * state.weather.intensity;

        // 12 Tied Physics Bands
        let tin_sound = self.filterbank.filters[0].process(rain_drive) * state.surfaces.tin * 2.2;
        let leaf_sound = self.filterbank.filters[1].process(rain_drive) * state.surfaces.leaves_broad * 1.5;
        let pine_sound = self.filterbank.filters[2].process(rain_drive) * state.surfaces.pine_needles * 1.7;
        let pavement_sound = self.filterbank.filters[3].process(rain_drive) * state.surfaces.pavement * 1.1;
        let water_sound = self.filterbank.filters[4].process(rain_drive) * state.surfaces.water_deep * 1.6;
        let puddle_sound = self.filterbank.filters[5].process(rain_drive) * state.surfaces.puddle_shallow * 1.8;
        let canvas_sound = self.filterbank.filters[6].process(rain_drive) * state.surfaces.canvas_tent * 1.9;
        let glass_sound = self.filterbank.filters[7].process(rain_drive) * state.surfaces.glass_window * 1.8;
        let wood_sound = self.filterbank.filters[8].process(rain_drive) * state.surfaces.wood_deck * 1.6;

        // Acoustic runoff & bubble chirps
        let runoff_drive = rain_drive * (state.surfaces.tin * 0.6 + state.surfaces.puddle_shallow * 0.4);
        let downpipe_sound = self.filterbank.filters[9].process(runoff_drive) * 1.3;

        // Discrete rain droplet Poisson impacts
        let droplet_prob = (state.weather.intensity * 0.04).clamp(0.001, 0.2);
        let droplet_burst = if self.rng.next_unit_f32() < droplet_prob {
            let droplet_pitch = 0.5 + self.rng.next_unit_f32() * 0.5;
            self.rng.next_f32() * droplet_pitch * 0.35
        } else {
            0.0
        };

        let bubble_sound = self.filterbank.filters[10].process(droplet_burst)
            * (state.surfaces.puddle_shallow + state.surfaces.water_deep)
            * 1.5;

        // Wind drive & howl band
        let wind_drive = self.brown_filter.process(white) * state.wind.speed;
        let wind_howl = self.filterbank.filters[11].process(wind_drive) * state.wind.howl * 2.0;

        // 4 Untied Residual Texture Bands
        let mist_drive = white * (state.weather.intensity * 0.2 + state.wind.speed * 0.1);
        let mist_sound = self.filterbank.filters[12].process(mist_drive) * 0.8;

        let turb_drive = wind_drive * (1.0 + state.wind.gustiness * 0.8) * 0.5;
        let turb_sound = self.filterbank.filters[13].process(turb_drive) * 0.9;

        let rattle_drive = rain_drive * (state.surfaces.leaves_broad + state.surfaces.pine_needles) * (state.wind.speed * 0.5 + 0.3);
        let rattle_sound = self.filterbank.filters[14].process(rattle_drive) * 0.7;

        let transducer_drive = white * (state.weather.intensity * 0.05);
        let transducer_sound = self.filterbank.filters[15].process(transducer_drive) * 0.6;

        let total_rain = tin_sound
            + leaf_sound
            + pine_sound
            + pavement_sound
            + water_sound
            + puddle_sound
            + canvas_sound
            + glass_sound
            + wood_sound
            + downpipe_sound
            + bubble_sound
            + mist_sound
            + rattle_sound
            + transducer_sound;

        let total_wind = (wind_drive * 0.6 + wind_howl + turb_sound) * (1.0 + state.wind.gustiness * 0.5);

        // 5. Side Sounds (Spatialized Point Sources)
        let mut side_w = 0.0;
        let mut side_x = 0.0;
        let mut side_y = 0.0;
        let mut side_z = 0.0;

        // Fireplace crackle (localized point source)
        if state.side_sounds.fireplace_intensity > 0.01 {
            let crackle_prob = state.side_sounds.fireplace_crackle_rate * 0.008;
            let crackle = if self.rng.next_unit_f32() < crackle_prob {
                self.rng.next_f32() * state.side_sounds.fireplace_intensity * 0.8
            } else {
                0.0
            };

            let azim = state.side_sounds.fireplace_azimuth;
            let elev = state.side_sounds.fireplace_elevation;
            let (cos_e, sin_e) = (elev.cos(), elev.sin());
            let (sin_a, cos_a) = azim.sin_cos();

            side_w += crackle * 0.7071;
            side_x += crackle * cos_e * cos_a;
            side_y += crackle * cos_e * sin_a;
            side_z += crackle * sin_e;
        }

        // Thunder low-frequency rumble
        if state.side_sounds.thunder_proximity > 0.05 {
            if self.rng.next_unit_f32() < 0.00005 * state.side_sounds.thunder_proximity {
                self.thunder_rumble = 0.9 * state.side_sounds.thunder_proximity;
            }
            if self.thunder_rumble > 0.001 {
                self.thunder_rumble *= self.thunder_decay;
                let thunder_noise = self.brown_filter.process(white) * self.thunder_rumble;
                
                let azim = state.side_sounds.thunder_azimuth;
                let elev = state.side_sounds.thunder_elevation;
                let (cos_e, sin_e) = (elev.cos(), elev.sin());
                let (sin_a, cos_a) = azim.sin_cos();

                side_w += thunder_noise * 0.7071;
                side_x += thunder_noise * cos_e * cos_a;
                side_y += thunder_noise * cos_e * sin_a;
                side_z += thunder_noise * sin_e;
            }
        }

        // Insects (Cicadas / Crickets granular chirp modulation)
        if state.side_sounds.insect_density > 0.01 {
            self.insect_phase = (self.insect_phase + 5200.0 / self.sample_rate).fract();
            let grain_env = ((self.insect_phase * 208.0).fract() * std::f32::consts::PI).sin().max(0.0);
            let carrier = (self.insect_phase * 2.0 * std::f32::consts::PI).sin();
            let noise_burst = self.rng.next_f32() * 0.15;
            let insect_sig = (carrier * 0.85 + noise_burst) * grain_env * state.side_sounds.insect_density * 0.35;

            let azim = state.side_sounds.insect_azimuth * std::f32::consts::PI;
            let dist = state.side_sounds.insect_proximity.clamp(0.2, 1.0);
            let atten = (1.0 / (dist * dist)).min(2.0);
            let (sin_a, cos_a) = azim.sin_cos();

            side_w += insect_sig * 0.7071 * atten;
            side_x += insect_sig * cos_a * atten;
            side_y += insect_sig * sin_a * atten;
        }

        // Birds (Randomized FM whistled chirps)
        if state.side_sounds.bird_activity > 0.01 {
            if self.bird_chirp_counter == 0 {
                let chirp_prob = state.side_sounds.bird_activity * 0.00008;
                if self.rng.next_unit_f32() < chirp_prob {
                    self.bird_chirp_counter = (self.sample_rate * 0.12) as usize; // 120ms
                    self.bird_chirp_duration = self.bird_chirp_counter.max(1);
                    self.bird_pitch = 3000.0 + self.rng.next_unit_f32() * 1200.0;
                }
            }

            if self.bird_chirp_counter > 0 {
                let progress = 1.0 - (self.bird_chirp_counter as f32 / self.bird_chirp_duration as f32);
                let env = (progress * std::f32::consts::PI).sin();
                let f = self.bird_pitch * (1.0 - progress * 0.25);
                let chirp = (progress * f * 2.0 * std::f32::consts::PI / self.sample_rate).sin() * env * 0.35;
                self.bird_chirp_counter -= 1;

                let dist = state.side_sounds.bird_proximity.clamp(0.2, 1.0);
                let atten = (1.0 / dist).min(1.5);
                let azim = -0.4 * std::f32::consts::PI;
                let (sin_a, cos_a) = azim.sin_cos();

                side_w += chirp * 0.7071 * atten;
                side_x += chirp * cos_a * atten;
                side_y += chirp * sin_a * atten;
                side_z += chirp * 0.35 * atten;
            }
        }

        // Wet Road Traffic (Low-mid highway Doppler spray whoosh)
        if state.side_sounds.traffic_distance > 0.01 {
            let dist = state.side_sounds.traffic_distance.clamp(0.05, 1.0);
            let traffic_white = white * 0.4;
            let spray = self.traffic_filter.process(traffic_white);
            let dist_atten = (1.0 - dist * 0.75).max(0.08);
            let traffic_sig = spray * dist_atten * 0.65;

            let azim = 0.8 * std::f32::consts::PI;
            let (sin_a, cos_a) = azim.sin_cos();

            side_w += traffic_sig * 0.7071;
            side_x += traffic_sig * cos_a * 0.5;
            side_y += traffic_sig * sin_a * 0.5;
        }

        // Master mixing into FOA B-Format:
        // W: omnidirectional energy
        // X: front-back
        // Y: left-right
        // Z: elevation (rain droplets falling from overhead + pitch angle)
        let rain_master = (total_rain + droplet_burst) * state.master_volume;
        let wind_master = total_wind * state.master_volume;

        let w = rain_master * 0.7071 + wind_master * 0.5 + side_w;
        let x = wind_master * 0.4 + side_x + (self.rng.next_f32() * 0.02 * rain_master);
        let y = (self.rng.next_f32() * 0.05 * rain_master) + side_y;
        // Rain falls from above, giving positive Z elevation component
        let z = (rain_master * 0.45 * state.weather.pitch_angle.cos()) + side_z;

        FoaFrame::new(w, x, y, z)
    }

    /// Process an entire buffer of FOA frames
    pub fn process_buffer(&mut self, state: &RainState, output: &mut [FoaFrame]) {
        for frame in output.iter_mut() {
            *frame = self.process_frame(state);
        }
    }
}
