#ifndef A2D02B60_26EF_4C91_9C76_865D41DA4F60
#define A2D02B60_26EF_4C91_9C76_865D41DA4F60

#define CREATE_CPU_TIMER                                    \
  std::chrono::time_point<std::chrono::steady_clock> a_ct1; \
  std::chrono::time_point<std::chrono::steady_clock> b_ct1;

#define START_CPU_TIMER a_ct1 = std::chrono::steady_clock::now();

#define STOP_CPU_TIMER b_ct1 = std::chrono::steady_clock::now();

#define TIME_ELAPSED_CPU_TIMER(PTR_MILLISEC) \
  ms = std::chrono::duration<float, std::milli>(b_ct1 - a_ct1).count();

#endif /* A2D02B60_26EF_4C91_9C76_865D41DA4F60 */
