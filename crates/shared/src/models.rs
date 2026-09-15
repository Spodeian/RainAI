//! Domain entities and collection management logic.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Priority {
    Low,
    #[default]
    Medium,
    High,
}

impl Priority {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub completed: bool,
}

impl Item {
    pub fn new(
        id: u64,
        title: impl Into<String>,
        description: impl Into<String>,
        priority: Priority,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            description: description.into(),
            priority,
            completed: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct ItemCollection {
    pub items: Vec<Item>,
    pub next_id: u64,
}

impl Default for ItemCollection {
    fn default() -> Self {
        let mut collection = Self {
            items: Vec::new(),
            next_id: 1,
        };
        collection.load_samples();
        collection
    }
}

impl ItemCollection {
    pub fn load_samples(&mut self) {
        self.items = vec![
            Item::new(
                1,
                "Explore Cross-Platform State Persistence",
                "State is automatically saved across browser sessions and desktop restarts.",
                Priority::High,
            ),
            Item::new(
                2,
                "Test Theme Toggle",
                "Switch between dark and warm light modes with high contrast readability.",
                Priority::Medium,
            ),
            Item::new(
                3,
                "Verify Offline PWA Capabilities",
                "Service workers cache static assets for offline usage on mobile and web.",
                Priority::Low,
            ),
        ];
        self.next_id = 4;
    }

    pub fn add(
        &mut self,
        title: impl Into<String>,
        description: impl Into<String>,
        priority: Priority,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.items.push(Item::new(id, title, description, priority));
        id
    }

    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.items.iter().position(|item| item.id == id) {
            self.items.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn toggle(&mut self, id: u64) {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == id) {
            item.completed = !item.completed;
        }
    }

    pub fn clear_completed(&mut self) {
        self.items.retain(|item| !item.completed);
    }

    pub fn reset(&mut self) {
        self.items.clear();
        self.next_id = 1;
    }

    pub fn completed_count(&self) -> usize {
        self.items.iter().filter(|i| i.completed).count()
    }

    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    pub fn completion_ratio(&self) -> f32 {
        if self.items.is_empty() {
            0.0
        } else {
            self.completed_count() as f32 / self.items.len() as f32
        }
    }
}
