#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include "__io.hh"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "<program> <datum name> <len>" << endl;
        exit(1);
    }
    string fname(argv[1]);
    size_t len     = atoi(argv[2]);
    auto   bincode = io::ReadBinaryFile<uint16_t>(fname, len);

    auto hist = new size_t[1024]();
    for_each(bincode, bincode + len, [&](auto& i) { hist[i]++; });
    for (size_t i = 0; i < 1024; i++) {
        if (hist[i] == 0) continue;
        cout << i << ": " << hist[i] << "\t" << endl;
    }

    return 0;
}
