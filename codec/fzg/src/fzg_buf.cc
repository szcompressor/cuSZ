#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fzg_hl.hh"
#include "mem/cxx_backends.h"
#include "mem/plan.h"

struct fzg::Buf2::impl : _ptb::buf_base {
  using Header = fzg_header;
  using E = uint16_t;
  using InputT = uint16_t;

  enum : int { ARCHIVE, COMP_LEN, SIGNUM, OFFSET_COUNTER, DECODED };

  static_assert(sizeof(bool) == 1, "SIGNUM is declared as U1");

  static constexpr size_t UINT32_BIT_LEN = 32;
  static constexpr size_t BLOCK_SIZE = 16;
  static constexpr size_t PAGE_BYTES = 4096;

  struct dims {
    size_t len;
    size_t data_bytes;  // padded
    size_t pad_len;     // padded
    size_t chunk_sz;
    size_t grid_x;
    size_t max_archive_bytes;
  };

  dims const d;

  GPU_unique_dptr<uint8_t[]> d_data, d_state;

  static size_t align_data_bytes(size_t data_bytes)
  {
    if (data_bytes == 0) return 0;
    data_bytes = (data_bytes - 1) / PAGE_BYTES + 1;
    return data_bytes * PAGE_BYTES;
  }

  static size_t compute_chunk_size(size_t data_bytes)
  {
    size_t denom = BLOCK_SIZE * UINT32_BIT_LEN;
    return (data_bytes + denom - 1) / denom;
  }

  static constexpr auto compute_grid_x = [](size_t data_bytes) { return data_bytes / PAGE_BYTES; };

  static size_t compute_max_archive_bytes(size_t chunk_sz, size_t grid_x, size_t len)
  {
    return sizeof(Header)                 // field 1: header
           + sizeof(uint32_t) * chunk_sz  // field 2: bitflag
           + sizeof(uint32_t) * grid_x    // field 3: start pos
           + sizeof(InputT) * len;        // field 4: max compressed output
  }

  static dims derive(size_t data_len)
  {
    auto const db = align_data_bytes(data_len * sizeof(InputT));
    auto const cs = compute_chunk_size(db);
    auto const gx = compute_grid_x(db);
    return {data_len, db, db / sizeof(InputT),
            cs,       gx, compute_max_archive_bytes(cs, gx, data_len)};
  }

  static tables plan(dims const& x, bool is_comp)
  {
    auto const enc = [is_comp](size_t n) { return is_comp ? n : 0; };
    return {
        {{ARCHIVE, U1, enc(x.max_archive_bytes)},
         {COMP_LEN, U4, enc(x.grid_x)},
         {SIGNUM, U1, enc(x.pad_len)},
         {DECODED, U2, is_comp ? 0 : x.pad_len}},
        {{OFFSET_COUNTER, U4, enc(1)}}};
  }

  impl(size_t data_len, bool is_comp) : impl(derive(data_len), is_comp) {}
  impl(dims const& x, bool is_comp) : buf_base(plan(x, is_comp)), d(x) {}

  ~impl() = default;

  void init() override
  {
    d_data = MAKE_UNIQUE_DEVICE(uint8_t, data().bytes());
    d_state = MAKE_UNIQUE_DEVICE(uint8_t, state().bytes());
    attach(d_data.get(), d_state.get());
  }

  uint8_t* archive() const { return data().ptr_of<uint8_t>(ARCHIVE); }
  uint32_t* comp_len() const { return data().ptr_of<uint32_t>(COMP_LEN); }
  bool* signum() const { return data().ptr_of<bool>(SIGNUM); }
  uint32_t* offset_counter() const { return state().ptr_of<uint32_t>(OFFSET_COUNTER); }
  E* decoded() const { return data().ptr_of<E>(DECODED); }

  uint32_t* bitflag() const
  {
    auto const b = archive();
    return b ? reinterpret_cast<uint32_t*>(b + sizeof(Header)) : nullptr;
  }

  uint32_t* start_pos() const
  {
    auto const b = archive();
    return b ? reinterpret_cast<uint32_t*>(b + sizeof(Header) + sizeof(uint32_t) * d.chunk_sz)
             : nullptr;
  }

  uint8_t* comp_out() const
  {
    auto const b = archive();
    return b ? b + sizeof(Header) + sizeof(uint32_t) * d.chunk_sz + sizeof(uint32_t) * d.grid_x
             : nullptr;
  }

  // for now, per-variable reset
  void reset(void* stream) override
  {
    if (offset_counter()) memset_device_async(offset_counter(), 1, 0, stream);
  }

  void memcpy_merge(Header& header, void* stream)
  {
    // layout ref.: [header | bitflag | start_pos | comp_out]
    cudaMemcpyAsync(
        archive(), &header, sizeof(Header), cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream));
  }

  [[deprecated]] void clear_buffer()
  {
    memset_device(archive(), d.max_archive_bytes);
    memset_device(offset_counter(), 1);
    memset_device(comp_len(), d.grid_x);
    memset_device(signum(), d.pad_len);
  }
};

fzg::Buf2::Buf2(size_t data_len, bool is_comp) : pimpl(std::make_unique<impl>(data_len, is_comp))
{
}

fzg::Buf2::~Buf2() = default;

void fzg::Buf2::init() { pimpl->init(); }
void fzg::Buf2::reset(void* stream) { pimpl->reset(stream); }

// or the owner allocates these itself and binds directly
size_t fzg::Buf2::planned_data_bytes() const { return pimpl->data().bytes(); }
size_t fzg::Buf2::planned_state_bytes() const { return pimpl->state().bytes(); }

void fzg::Buf2::set_base(void* d_data, void* d_state) { pimpl->set_base(d_data, d_state); }
void fzg::Buf2::attach(void* d_data, void* d_state) { pimpl->attach(d_data, d_state); }

size_t fzg::Buf2::len() const { return pimpl->d.len; }
size_t fzg::Buf2::pad_len() const { return pimpl->d.pad_len; }
size_t fzg::Buf2::data_bytes() const { return pimpl->d.data_bytes; }
size_t fzg::Buf2::chunk_size() const { return pimpl->d.chunk_sz; }
size_t fzg::Buf2::grid_x() const { return pimpl->d.grid_x; }
size_t fzg::Buf2::archive_bytes() const { return pimpl->d.max_archive_bytes; }
size_t fzg::Buf2::archive_bytes(size_t data_len)
{ return impl::derive(data_len).max_archive_bytes; }

// device pointers
uint32_t* fzg::Buf2::bitflag_d() const { return pimpl->bitflag(); }
uint32_t* fzg::Buf2::start_pos_d() const { return pimpl->start_pos(); }
uint8_t* fzg::Buf2::comp_out_d() const { return pimpl->comp_out(); }
uint8_t* fzg::Buf2::archive_d() const { return pimpl->archive(); }
uint32_t* fzg::Buf2::comp_len_d() const { return pimpl->comp_len(); }
uint32_t* fzg::Buf2::offset_counter_d() const { return pimpl->offset_counter(); }
bool* fzg::Buf2::signum_d() const { return pimpl->signum(); }
fzg::E* fzg::Buf2::out_d() const { return pimpl->decoded(); }

// ops
void fzg::Buf2::clear_buffer() { pimpl->clear_buffer(); }
void fzg::Buf2::memcpy_merge(Header& header, void* stream) { pimpl->memcpy_merge(header, stream); }
