//! KV-cache region with descriptor-only compaction.

use crate::planner::TensorOffset;

/// One slot in the KV cache: a descriptor pointing at data in the arena. The
/// payload lives in the arena; this is only the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSlot {
    pub token_index: u32,
    pub offset: TensorOffset,
    pub valid: bool,
}

/// The contiguous KV-cache, addressed through descriptors. Compaction shuffles
/// descriptors only — payload bytes are never moved or copied (ADR-003).
#[derive(Debug, Default)]
pub struct KvRegion {
    slots: Vec<KvSlot>,
}

impl KvRegion {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn len(&self) -> u32 {
        self.slots.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Append a committed token's descriptor.
    pub fn push(&mut self, offset: TensorOffset) {
        let token_index = self.slots.len() as u32;
        self.slots.push(KvSlot {
            token_index,
            offset,
            valid: true,
        });
    }

    /// Mark a slot pruned (e.g. rejected draft / compressed-away token).
    pub fn mark_pruned(&mut self, token_index: u32) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.token_index == token_index) {
            s.valid = false;
        }
    }

    pub fn slots(&self) -> &[KvSlot] {
        &self.slots
    }

    /// Drop tail descriptors so the region retains only the first `len` slots.
    /// `O(dropped)`; survivors keep their `offset` (no data copy). This is the
    /// rollback primitive for the safety control loop (ADR-012): restoring a
    /// checkpoint rewinds committed KV without replaying prefill.
    pub fn truncate(&mut self, len: u32) {
        self.slots.truncate(len as usize);
    }

    /// Remove pruned descriptors and re-index survivors. Returns how many were
    /// reclaimed. **Survivors keep their original `offset`** — proof that data
    /// is not moved, only descriptors are reshuffled.
    pub fn compact(&mut self) -> u32 {
        let before = self.slots.len();
        self.slots.retain(|s| s.valid);
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.token_index = i as u32;
        }
        (before - self.slots.len()) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_is_descriptor_only() {
        let mut kv = KvRegion::new();
        kv.push(0); // token 0 @ off 0
        kv.push(64); // token 1 @ off 64
        kv.push(128); // token 2 @ off 128
        kv.mark_pruned(1);

        let off_token2_before = kv.slots()[2].offset;
        let reclaimed = kv.compact();

        assert_eq!(reclaimed, 1);
        assert_eq!(kv.len(), 2);
        // Survivor's payload offset is unchanged (no data copy)...
        assert_eq!(kv.slots()[1].offset, off_token2_before);
        // ...but its logical index was shuffled down.
        assert_eq!(kv.slots()[1].token_index, 1);
    }

    #[test]
    fn truncate_rewinds_to_len_without_touching_survivors() {
        let mut kv = KvRegion::new();
        kv.push(0);
        kv.push(64);
        kv.push(128);
        let off1 = kv.slots()[1].offset;
        kv.truncate(2);
        assert_eq!(kv.len(), 2);
        assert_eq!(kv.slots()[1].offset, off1); // survivor untouched (no data copy)
        kv.truncate(0);
        assert!(kv.is_empty());
    }
}
