#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

// ─── Alignment helpers ──────────────────────────────────────────────
// 64-byte base alignment improves the compiler's ability to issue wide loads/stores
// on hot SoA arrays.
static constexpr std::size_t kSoAAlign = 64;
static_assert((kSoAAlign & (kSoAAlign - 1)) == 0, "kSoAAlign must be a power of two");
static_assert(kSoAAlign >= alignof(std::max_align_t),
              "kSoAAlign must satisfy max_align_t alignment");

inline std::size_t soa_align_up(std::size_t offset, std::size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

template <typename T>
inline constexpr std::size_t soa_hot_alignment_v =
    (kSoAAlign > alignof(T) ? kSoAAlign : alignof(T));

// ─── SoABuffer ──────────────────────────────────────────────────────
// Owns a single contiguous allocation.  SoA structs use this as a
// member and point their raw `T*` fields into it.
//
// Not copyable directly — the owning SoA struct implements copy by
// allocating a new SoABuffer and doing a single memcpy + pointer fixup.

struct SoABuffer {
  struct FreeDeleter {
    void operator()(void *p) const { std::free(p); }
  };

  using Ptr = std::unique_ptr<void, FreeDeleter>;
  Ptr data{nullptr};
  std::size_t byte_size{0};

  SoABuffer() = default;

  static SoABuffer allocate(std::size_t bytes) {
    SoABuffer buf;
    buf.byte_size = soa_align_up(bytes, kSoAAlign);
    if (buf.byte_size > 0) {
      void *p = std::aligned_alloc(kSoAAlign, buf.byte_size);
      if (!p) throw std::bad_alloc();
      std::memset(p, 0, buf.byte_size);
      buf.data.reset(p);
    }
    return buf;
  }

  // Deep copy: allocate + memcpy.
  [[nodiscard]] SoABuffer deep_copy() const {
    SoABuffer copy;
    copy.byte_size = byte_size;
    if (byte_size > 0) {
      void *p = std::aligned_alloc(kSoAAlign, byte_size);
      if (!p) throw std::bad_alloc();
      std::memcpy(p, data.get(), byte_size);
      copy.data.reset(p);
    }
    return copy;
  }

  [[nodiscard]] char *base() { return reinterpret_cast<char *>(data.get()); }
  [[nodiscard]] const char *base() const {
    return reinterpret_cast<const char *>(data.get());
  }

  SoABuffer(SoABuffer &&) noexcept = default;
  SoABuffer &operator=(SoABuffer &&) noexcept = default;
  SoABuffer(const SoABuffer &) = delete;
  SoABuffer &operator=(const SoABuffer &) = delete;
};

// ─── Layout helper ──────────────────────────────────────────────────
// Computes byte offsets for fields laid out sequentially in a buffer.
//
// Usage (in SoA struct):
//   SoALayout layout;
//   layout.begin();
//   std::size_t off_state = layout.add_field<uint8_t>(n);
//   std::size_t off_flags = layout.add_field<uint8_t>(n);
//   std::size_t off_device = layout.add_field<int32_t>(n);
//   ...
//   SoABuffer buf = SoABuffer::allocate(layout.total());
//   state = reinterpret_cast<uint8_t*>(buf.base() + off_state);
//   ...

struct SoALayout {
  std::size_t offset{0};

  void begin() { offset = 0; }

  std::size_t add_bytes(std::size_t bytes, std::size_t alignment) {
    offset = soa_align_up(offset, alignment);
    const std::size_t result = offset;
    offset += bytes;
    return result;
  }

  template <typename T>
  std::size_t add_field(std::size_t count) {
    return add_bytes(sizeof(T) * count, alignof(T));
  }

  template <typename T>
  std::size_t add_hot_field(std::size_t count) {
    return add_bytes(sizeof(T) * count, soa_hot_alignment_v<T>);
  }

  [[nodiscard]] std::size_t total() const { return offset; }
};

// ─── soa_ptr_at ─────────────────────────────────────────────────────
// Cast base+offset to T* and assert alignment to the compiler.
// Default alignment is alignof(T); override for hot fields.
template <typename T, std::size_t Alignment = alignof(T)>
[[nodiscard]] T * __restrict__ soa_ptr_at(char *base, std::size_t byte_offset) noexcept {
  static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
  static_assert(Alignment >= alignof(T), "Alignment must satisfy T alignment");
  return std::assume_aligned<Alignment>(
      reinterpret_cast<T *>(base + byte_offset));
}
