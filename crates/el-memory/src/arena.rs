//! The single contiguous arena, allocated once.

/// A contiguous byte buffer allocated once at session init. The decode loop
/// borrows sub-slices at planned offsets and never allocates (ADR-003).
///
/// Note: true OS page-alignment is a platform concern handled when wiring a real
/// engine/DMA path; this safe-Rust buffer models the allocate-once + fixed-offset
/// contract that the planner depends on.
#[derive(Debug)]
pub struct Arena {
    buf: Box<[u8]>,
}

impl Arena {
    /// Allocate the arena once. This is the *only* large allocation per session.
    pub fn new(size: usize) -> Self {
        Self {
            buf: vec![0u8; size].into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Borrow a planned region immutably. Returns `None` if out of bounds.
    pub fn region(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(usize::try_from(len).ok()?)?;
        self.buf.get(start..end)
    }

    /// Borrow a planned region mutably. Returns `None` if out of bounds.
    pub fn region_mut(&mut self, offset: u64, len: u64) -> Option<&mut [u8]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(usize::try_from(len).ok()?)?;
        self.buf.get_mut(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_and_borrows_regions() {
        let mut a = Arena::new(1024);
        assert_eq!(a.len(), 1024);
        {
            let r = a.region_mut(16, 8).unwrap();
            r.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        }
        assert_eq!(a.region(16, 8).unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.region(1020, 8).is_none(), "out of bounds yields None");
    }
}
