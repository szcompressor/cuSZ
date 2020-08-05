#include <bitset>
#include <iostream>

using namespace std;

template <int Magnitude>
unsigned int NextButterflyIdx(unsigned int idx, unsigned int iter)
{
    auto team_size        = 1 << (iter - 1);
    auto next_team_size   = 2 * team_size;
    auto league_size      = 1 << (Magnitude - iter + 1);
    auto next_league_size = league_size / 2;

    auto team_rank           = idx % team_size;
    auto league_rank         = idx / team_size;
    auto next_subleague_rank = league_rank / next_league_size;
    auto next_league_rank    = league_rank % next_league_size;
    auto next_rank           = next_league_rank * next_team_size + next_subleague_rank * team_size + team_rank;

    return next_rank;
}

int main()
{
    auto src = new int[16];
    auto dst = new int[16];
    int* ex;
    /*
        for (auto i = 0; i < 16; i++) cout << i << "->" << NextButterflyIdx<4>(i, 1) << endl;
        cout << endl;
        for (auto i = 0; i < 16; i++) cout << i << "->" << NextButterflyIdx<4>(i, 2) << endl;
        cout << endl;
        for (auto i = 0; i < 16; i++) cout << i << "->" << NextButterflyIdx<4>(i, 3) << endl;
    */
    auto a = NextButterflyIdx<4>(10, 1);
    cout << a << endl;
    auto b = NextButterflyIdx<4>(a, 2);
    cout << b << endl;
    auto c = NextButterflyIdx<4>(b, 3);
    cout << c << endl;
}
