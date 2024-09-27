#include <include/queues.hpp>
#include <queue>

int main() {
  Randomizer<int, std::priority_queue> queue;
  queue.push_random(5);
  queue.push_random(1);
  queue.push_random(3);
  queue.push_random(2);
  queue.push_random(4);

  print_table(queue);

  return 0;
}