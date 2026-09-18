use audio::engine::*;

#[test]
fn test_audio_ring_buffer_fifo() {
    let mut rb = AudioRingBuffer::new(4);
    assert_eq!(rb.available_frames(), 0);
    assert_eq!(rb.free_frames(), 4);

    assert!(rb.push_frame(0.1, 0.2));
    assert!(rb.push_frame(0.3, 0.4));
    assert_eq!(rb.available_frames(), 2);

    let (l1, r1) = rb.pop_frame().unwrap();
    assert!((l1 - 0.1).abs() < 1e-6);
    assert!((r1 - 0.2).abs() < 1e-6);

    assert!(rb.push_frame(0.5, 0.6));
    assert!(rb.push_frame(0.7, 0.8));
    assert!(rb.push_frame(0.9, 1.0));
    assert_eq!(rb.available_frames(), 4);
    assert!(!rb.push_frame(1.1, 1.2));

    let (l2, r2) = rb.pop_frame().unwrap();
    assert!((l2 - 0.3).abs() < 1e-6);
    assert!((r2 - 0.4).abs() < 1e-6);
}

#[test]
fn test_master_soft_limiter_saturation() {
    assert_eq!(soft_limit(0.0), 0.0);
    assert_eq!(soft_limit(0.5), 0.5);
    assert_eq!(soft_limit(-0.5), -0.5);

    let saturated_pos = soft_limit(2.5);
    let saturated_neg = soft_limit(-2.5);
    assert!(saturated_pos < 1.0);
    assert!(saturated_pos > 0.9);
    assert!(saturated_neg > -1.0);
    assert!(saturated_neg < -0.9);

    let extreme = soft_limit(100.0);
    assert!(extreme <= 1.0);
    assert!(extreme > 0.999);
}
