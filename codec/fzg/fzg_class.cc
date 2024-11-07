#include "fzg_class.hh"

#include "fzg_type.h"
#include "kernel/fzg_cx.hh"

struct psz::FzgCodec::impl {
 public:
  using Header = fzg_header;
  using M = uint32_t;

  Header header;
  fzgpu::config_map config;
  fzgpu::Buf* buf;

  size_t len;

  float _time_lossless{0.0};
  float time_codec() const { return _time_lossless; }

  impl(size_t const data_len)
  {
    len = data_len;
    config =
        fzgpu::configure_fzgpu(data_len);  // TODO duplicate with Buf::config

    buf = new fzgpu::Buf(data_len, false);
  }

  ~impl() { delete buf; }

  void make_metadata()
  {
    header.original_len = len;

    M nbyte[FZGHEADER_END];
    nbyte[FZGHEADER_HEADER] = TODO_CHANGE_FZGHEADER_SIZE;
    nbyte[FZGHEADER_BITFLAG] = sizeof(uint32_t) * config.at("chunk_size");
    nbyte[FZGHEADER_START_POS] = sizeof(uint32_t) * config.at("grid_x");
    nbyte[FZGHEADER_BITSTREAM] = buf->h_offset_sum * sizeof(uint32_t);

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < FZGHEADER_END + 1; i++)
      header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < FZGHEADER_END + 1; i++)
      header.entry[i] += header.entry[i - 1];
  }

  void encode(
      E* in_data, size_t const data_len, uint8_t** out_archive,
      size_t* archive_len, void* stream)
  {
    fzgpu::cuhip::GPU_FZ_encode(
        in_data, data_len, buf->d_offset_counter, buf->d_bitflag_array,
        buf->d_start_pos, buf->d_comp_out, buf->d_comp_len,
        (cudaStream_t)stream);

    cudaStreamSynchronize((cudaStream_t)stream);

    cudaMemcpy(
        &(buf->h_offset_sum), buf->d_offset_counter, sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    // output
    make_metadata();
    cudaMemcpy(
        buf->d_archive, &header, sizeof(fzg_header), cudaMemcpyHostToDevice);
    *out_archive = buf->d_archive;
    *archive_len = (size_t)header.entry[FZGHEADER_END];
  }

  void decode(
      uint8_t* in_archive, size_t const archive_len, E* out_data,
      size_t const data_len, void* stream)
  {
    Header header;
    cudaMemcpy(&header, in_archive, sizeof(header), cudaMemcpyDeviceToHost);

#define FZG_ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_archive + header.entry[FZGHEADER_##SYM])

    // fzgpu::cuhip::GPU_FZ_decode(
    //     in_comp, in_bitflag, in_start_pos, out_data, data_len,
    //     (cudaStream_t)stream);
    fzgpu::cuhip::GPU_FZ_decode(
        FZG_ACCESSOR(BITSTREAM, uint8_t), FZG_ACCESSOR(BITFLAG, uint32_t),
        FZG_ACCESSOR(START_POS, uint32_t), out_data, data_len,
        (cudaStream_t)stream);

#undef PHF_ACCESSOR
  }
};

// link impl and front-end

psz::FzgCodec::FzgCodec(size_t const data_len) :
    pimpl{std::make_unique<impl>(data_len)}
{
}

psz::FzgCodec::~FzgCodec(){};

psz::FzgCodec* psz::FzgCodec::encode(
    E* in_data, size_t const data_len, uint8_t** out_comp, size_t* comp_len,
    void* stream)
{
  pimpl->encode(in_data, data_len, out_comp, comp_len, stream);
  return this;
}

psz::FzgCodec* psz::FzgCodec::decode(
    uint8_t* in_comp, size_t const comp_len, E* out_data,
    size_t const data_len, void* stream)
{
  pimpl->decode(in_comp, comp_len, out_data, data_len, (cudaStream_t)stream);
  return this;
}

size_t psz::FzgCodec::expose_padded_input_len() const
{
  return pimpl->config.at("pad_len");
}