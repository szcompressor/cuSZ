/**
 * @file bin_hfserial.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-17
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "cusz/type.h"
#include "hf/hf.hh"
#include "hf/hfbk.hh"
#include "hf/hfbk_impl.hh"
#include "hf/hfcanon.hh"
#include "hf/hfword.hh"
#include "mem/memseg_cxx.hh"

void printcode_u4(u4 idx, u4* word)
{
  using PW = PackedWordByWidth<4>;
  auto pw = (PW*)word;
  cout << idx << "\t"  //
       << bitset<PW::FIELDWIDTH_bits>(pw->bits) << " (" << pw->bits << ")\t"
       << bitset<PW::FIELDWIDTH_word>(pw->word) << "\n";
}

void hfbook_serial_reference(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");
  auto space = new hf_canon_reference<u4, u4>(bklen);

  hist->control({MallocHost})->file(fname.c_str(), FromFile);
  book->control({MallocHost});
  memset(book->hptr(), 0xff, sizeof(u4) * bklen);

  hf_buildtree_impl2<u4>(hist->hptr(), bklen, book->hptr());

  for (auto i = 0; i < bklen; i++) {
    auto res = book->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  space->icb() = book->hptr();  // external
  space->canonize();

  memcpy(book->hptr(), space->ocb(), sizeof(u4) * bklen);

  printf("serial 1, canonized (CPU): \n");
  for (auto i = 0; i < bklen; i++) {
    auto res = book->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  delete hist;
  delete space;
}

void hfbook_serial_integrated(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");

  auto revbook_bytes = hf_space<u4, u4>::revbook_bytes(bklen);
  auto revbook = new pszmem_cxx<u1>(revbook_bytes, 1, 1, "revbook");

  hist->control({MallocHost})->file(fname.c_str(), FromFile);
  book->control({MallocHost});
  revbook->control({MallocHost});

  psz::hf_buildbook<SEQ, u4, u4>(
      hist->hptr(), bklen, book->hptr(), revbook->hptr(), revbook_bytes,
      nullptr);

  for (auto i = 0; i < bklen; i++) {
    auto res = book->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  delete hist;
  delete book;
  delete revbook;
}

// for reference
void hfbook_gpu(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");

  hist->control({MallocHost, Malloc})
      ->file(fname.c_str(), FromFile)
      ->control({H2D});

  cusz::HuffmanCodec<u4, u4> codec;

  auto fakelen1 = bklen * 100;
  auto fakelen2 = 768;

  codec.init(fakelen1, bklen, fakelen2);
  codec.build_codebook(hist->dptr(), bklen, 0);

  codec.bk4->control({D2H});

  for (auto i = 0; i < bklen; i++) {
    auto res = codec.bk4->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("0     1           2\n");
    printf("PROG  fname-hist  booklen\n");

    return -1;
  }

  auto fname = std::string(argv[1]);
  auto bklen = atoi(argv[2]);

  cout << "previous working reference:" << endl;
  hfbook_serial_reference(fname, bklen);
  cout << "serial integrate:" << endl;
  hfbook_serial_integrated(fname, bklen);
  // cout << "GPU (reference):" << endl;
  // hfbook_gpu(fname, bklen);

  //
  return 0;
}