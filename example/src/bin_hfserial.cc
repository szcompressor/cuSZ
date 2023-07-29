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
#include "hf/hf_word.hh"
#include "hf/hfserial_book.hh"
#include "hf/hfserial_book2.hh"
#include "hf/hfserial_canon.hh"
#include "mem/memseg_cxx.hh"

void printcode_u4(u4 idx, u4* word)
{
  using PW = PackedWordByWidth<4>;
  auto pw = (PW*)word;
  cout << idx << "\t"  //
       << bitset<PW::field_bits>(pw->bits) << " (" << pw->bits << ")\t"
       << bitset<PW::field_word>(pw->word) << "\n";
}

void hfbook_serial1(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto raw_book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");
  auto space = new hf_canon_space<u4, u4>(bklen);

  hist->control({MallocHost})->file(fname.c_str(), FromFile);
  raw_book->control({MallocHost});
  memset(raw_book->hptr(), 0xff, sizeof(u4) * bklen);

  hf_create_book_serial<u4>(hist->hptr(), bklen, raw_book->hptr());

  for (auto i = 0; i < bklen; i++) {
    auto res = raw_book->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  space->icb() = raw_book->hptr();
  space->canonize();

  printf("serial 1, canonized (CPU): \n");
  for (auto i = 0; i < bklen; i++) {
    auto res = space->ocb(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  delete hist;
  delete space;
}

void hfbook_serial2(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto raw_book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");
  auto space = new hf_canon_space<u4, u4>(bklen);

  hist->control({MallocHost})->file(fname.c_str(), FromFile);
  raw_book->control({MallocHost});
  memset(raw_book->hptr(), 0xff, sizeof(u4) * bklen);

  hf_build_book2<u4>(hist->hptr(), bklen, raw_book->hptr());

  for (auto i = 0; i < bklen; i++) {
    auto res = raw_book->hptr(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  space->icb() = raw_book->hptr();
  space->canonize();

  printf("serial 2, canonized: \n");
  for (auto i = 0; i < bklen; i++) {
    auto res = space->ocb(i);
    if (res != 0 and res != 0xffffffff) printcode_u4(i, &res);
  }

  delete hist;
  delete space;
}

// for reference
void hfbook_gpu(string fname, int bklen)
{
  auto hist = new pszmem_cxx<u4>(bklen, 1, 1, "histogram");
  auto raw_book = new pszmem_cxx<u4>(bklen, 1, 1, "internal book");
  auto space = new hf_canon_space<u4, u4>(bklen);

  hist->control({MallocHost, Malloc})
      ->file(fname.c_str(), FromFile)
      ->control({H2D});

  cusz::HuffmanCodec<u4, u4, u4> codec;

  auto fakelen1 = bklen * 100;
  auto fakelen2 = 768;

  codec.init(fakelen1, bklen, fakelen2);
  codec.build_codebook(hist->dptr(), bklen, 0);

  codec.book->control({D2H});

  for (auto i = 0; i < bklen; i++) {
    auto res = codec.book->hptr(i);
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

  cout << "serial 1 (fail):" << endl;
  hfbook_serial1(fname, bklen);
  // cout << "serial 2:" << endl;
  // hfbook_serial2(fname, bklen);
  cout << "GPU (reference):" << endl;
  hfbook_gpu(fname, bklen);

  //
  return 0;
}