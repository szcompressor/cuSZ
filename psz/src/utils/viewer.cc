#include "utils/viewer/viewer.h"

#include <cstddef>

#include "tehm.hh"
#include "utils/viewer/viewer.noarch.hh"

double get_total_time(psz::timerecord_t r)
{
  double total = 0.0;
  std::for_each(r->begin(), r->end(), [&](psz::TimeRecordTuple t) {
    return total += std::get<1>(t);
  });
  return total;
}

void* psz_make_timerecord() { return (void*)new psz::TimeRecord; }

void psz_review_timerecord(void* _r, psz_header* h)
{
  auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
  auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();

  auto r = (psz::timerecord_t)_r;

  psz::TimeRecord reflow;

  {  // reflow
    psz::TimeRecordTuple book_tuple;

    auto total_time = get_total_time(r);
    auto subtotal_time = total_time;

    for (auto& i : *r) {
      auto item = std::string(std::get<0>(i));
      if (item == "book") {
        book_tuple = i;
        subtotal_time -= std::get<1>(i);
      }
      else {
        reflow.push_back(i);
      }
    }
    reflow.push_back({const_cast<const char*>("(subtotal)"), subtotal_time});
    printf("\e[2m");
    reflow.push_back(book_tuple);
    reflow.push_back({const_cast<const char*>("(total)"), total_time});
    printf("\e[0m");
  }

  psz_utils::println_throughput_tablehead();
  for (auto& i : reflow)
    psz_utils::println_throughput(
        std::get<0>(i), std::get<1>(i), uncomp_bytes);

  printf("\n");
}

void psz_review_cr(psz_header* h)
{
  // [TODO] put error status
  if (h->dtype != F4 and h->dtype != F8)
    cout << "[psz::log::fatal_error] original length is is zero." << endl;

  auto comp_bytes = [&]() {
    auto END = sizeof(h->entry) / sizeof(h->entry[0]);
    return h->entry[END - 1];
  };

  auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
  auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();
  auto fieldsize = [&](auto FIELD) {
    return h->entry[FIELD + 1] - h->entry[FIELD];
  };
  auto __print = [&](auto str, auto num) {
    cout << "  ";
    cout << std::left;
    cout << std::setw(28) << str;
    cout << std::right;
    cout << std::setw(10) << num;
    cout << '\n';
  };
  auto __print_perc = [&](auto str, auto num) {
    auto perc = num * 100.0 / comp_bytes();
    cout << "  ";
    cout << std::left;
    cout << std::setw(28) << str;
    cout << std::right;
    cout << std::setw(10) << num;
    cout << std::setw(10) << std::setprecision(3) << std::fixed << perc
         << "%\n";
  };
  auto __newline = []() { cout << '\n'; };

  if (comp_bytes() != 0) {
    auto cr = 1.0 * uncomp_bytes / comp_bytes();
    __newline();
    __print("psz::comp::review::CR", cr);
  }
  else {
    cout << "[psz::log::fatal_error] compressed len is zero." << endl;
  }

  __print("original::bytes", uncomp_bytes);
  __print("original::bytes", uncomp_bytes);
  __print("compressed::bytes", comp_bytes());
  __newline();
  __print_perc("compressed::total::bytes", comp_bytes());
  printf("  ------------------------\n");
  __print_perc("compressed::HEADER::bytes", sizeof(psz_header));
  __print_perc("compressed::ANCHOR::bytes", fieldsize(PSZHEADER_ANCHOR));
  __print_perc("compressed::VLE::bytes", fieldsize(PSZHEADER_VLE));
  __print_perc("compressed::SPFMT::bytes", fieldsize(PSZHEADER_SPFMT));
  __newline();
  __print(
      "compressed::ANCHOR:::len", fieldsize(PSZHEADER_ANCHOR) / sizeof_T());
  __print(
      "compressed::OUTLIER:::len",
      fieldsize(PSZHEADER_SPFMT) / (sizeof_T() + sizeof(uint32_t)));
}

void psz_review_compression(void* r, psz_header* h)
{
  printf("\n(c) COMPRESSION REPORT\n");
  psz_review_cr(h);
  psz_review_timerecord((psz::timerecord_t)r, h);
}

void psz_review_decompression(void* r, size_t bytes)
{
  printf("\n(d) deCOMPRESSION REPORT\n");

  auto total_time = get_total_time((psz::timerecord_t)r);
  (*(psz::timerecord_t)r)
      .push_back({const_cast<const char*>("(total)"), total_time});

  psz_utils::println_throughput_tablehead();
  for (auto& i : *(psz::timerecord_t)r)
    psz_utils::println_throughput(std::get<0>(i), std::get<1>(i), bytes);

  printf("\n");
}