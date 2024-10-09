#include <queue>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/queues.hpp"
#include <chrono>

TEST_CASE("TopKQueue: push") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);
  queue.push(2);
  queue.push(4);

  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 5);
  CHECK(queue.top() == 5);
}

TEST_CASE("TopKQueue: push: pop") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);
  queue.push(2);
  queue.push(4);

  queue.pop();
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 4);
  CHECK(queue.top() == 4);

  queue.pop();
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 3);
  CHECK(queue.top() == 3);
}

TEST_CASE("TopKQueue: top") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);

  CHECK(queue.top() == 5);
  queue.push(6);
  CHECK(queue.top() == 6);
}

TEST_CASE("TopKQueue: at") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);

  CHECK(queue.at(0) == 5);
  CHECK(queue.at(1) == 3);
  CHECK(queue.at(2) == 1);
}

TEST_CASE("TopKQueue: remove_at") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);
  queue.push(2);
  queue.push(4);

  queue.remove_at(1);
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 4);
  CHECK(queue.top() == 5);

  queue.remove_at(0);
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 3);
  CHECK(queue.top() == 3);
}

TEST_CASE("TopKQueue: remove") {
  Top3Queue queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);
  queue.push(2);
  queue.push(4);

  // Top k elemments of queue should be [5, 4, 3]
  std::vector<std::size_t> indices = {0, 2};
  queue.remove(indices);
  // Top k elements of queue should be [4, 2, 1]
  print(queue);
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 3);
  CHECK(queue.top() == 4);

  queue.push(10);
  queue.push(7);
  queue.push(6);

  // Top k elements of queue should be [10, 7, 6]
  indices = {1};
  queue.remove(indices);
  // Top k elements of queue should be [10, 6, 4]

  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 5);
  CHECK(queue.top() == 10);

  queue.pop();
  queue.pop();

  // Top k elements of queue should be [4, 2, 1]
  CHECK(queue.topk_size() == 3);
  CHECK(queue.size() == 3);
  CHECK(queue.top() == 4);

  indices = {0};
  queue.remove(indices);

  // Top k elements of queue should be [2, 1]

  CHECK(queue.topk_size() == 2);
  CHECK(queue.size() == 2);
  CHECK(queue.top() == 2);

  queue.pop();

  // Top k elements of queue should be [1]
  CHECK(queue.topk_size() == 1);
  CHECK(queue.size() == 1);
  CHECK(queue.top() == 1);
}

TEST_CASE("TopKQueue: empty") {
  Top3Queue queue;
  CHECK(queue.empty());

  queue.push(5);
  CHECK_FALSE(queue.empty());

  queue.pop();
  CHECK(queue.empty());

  queue.push(1);
  queue.push(3);
  queue.push(2);
  queue.push(4);

  queue.pop();
  queue.pop();
  queue.pop();
  queue.pop();
  CHECK(queue.empty());
}

TEST_CASE("TopKQueue: size") {
  Top3Queue queue;
  CHECK(queue.size() == 0);

  queue.push(5);
  queue.push(1);
  queue.push(3);
  CHECK(queue.size() == 3);

  queue.push(2);
  queue.push(4);
  CHECK(queue.size() == 5);
}

TEST_CASE("TopKQueue: equivalent") {
  Top3Queue queue1;
  std::priority_queue<int> queue2;

  queue1.push(5);
  queue1.push(1);
  queue1.push(3);
  queue1.push(2);
  queue1.push(4);
  queue1.push(10);
  queue1.push(7);

  queue2.push(5);
  queue2.push(1);
  queue2.push(3);
  queue2.push(2);
  queue2.push(4);
  queue2.push(10);
  queue2.push(7);

  for (int i = 0; i < 5; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }

  queue1.push(5);
  queue1.push(10);
  queue1.push(7);

  queue2.push(5);
  queue2.push(10);
  queue2.push(7);

  for (int i = 0; i < 3; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }
}
TEST_CASE("TopKQueue: container equivalent") {
  ContainerQueue<int, TopKQueueHelper<3>::queue_type> queue1;
  ContainerQueue<int, std::priority_queue> queue2;

  queue1.push(5, 5);
  queue1.push(1, 1);
  queue1.push(3, 3);
  queue1.push(2, 2);
  queue1.push(4, 4);
  queue1.push(10, 10);
  queue1.push(7, 7);

  queue2.push(5, 5);
  queue2.push(1, 1);
  queue2.push(3, 3);
  queue2.push(2, 2);
  queue2.push(4, 4);
  queue2.push(10, 10);
  queue2.push(7, 7);

  for (int i = 0; i < 5; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }

  queue1.push(5, 5);
  queue1.push(10, 10);
  queue1.push(7, 7);

  queue2.push(5, 5);
  queue2.push(10, 10);
  queue2.push(7, 7);

  for (int i = 0; i < 3; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }
}

TEST_CASE("TopKQueue: container equivalent (zero)") {
  ContainerQueue<int, TopKQueueHelper<3>::queue_type> queue1;
  ContainerQueue<int, std::priority_queue> queue2;

  queue1.push(5, 0);
  queue1.push(1, 0);
  queue1.push(3, 0);
  queue1.push(2, 0);
  queue1.push(4, 0);
  queue1.push(10, 0);
  queue1.push(7, 0);

  queue2.push(5, 0);
  queue2.push(1, 0);
  queue2.push(3, 0);
  queue2.push(2, 0);
  queue2.push(4, 0);
  queue2.push(10, 0);
  queue2.push(7, 0);

  for (int i = 0; i < 5; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }

  queue1.push(5, 0);
  queue1.push(10, 0);
  queue1.push(7, 0);

  queue2.push(5, 0);
  queue2.push(10, 0);
  queue2.push(7, 0);

  for (int i = 0; i < 3; i++) {
    CHECK(queue1.top() == queue2.top());
    queue1.pop();
    queue2.pop();
  }
}

TEST_CASE("TopKQueue: performance (asc)") {
  auto start = std::chrono::high_resolution_clock::now();

  Top10Queue queue;
  for (int i = 0; i < 1000000; i++) {
    queue.push(i);
  }
  for (int k = 0; k < 1000; k++) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 0; i < 10; i++) {
      queue.push(i);
    }
  }
}

TEST_CASE("TopKQueue: performance (desc)") {
  Top10Queue queue;
  for (int i = 1000000; i > 0; i--) {
    queue.push(i);
  }
  for (int k = 1000; k > 0; k--) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 10; i > 0; i--) {
      queue.push(i);
    }
  }
}

TEST_CASE("PriorityQueue: performance") {
  std::priority_queue<int> queue;
  for (int i = 0; i < 1000000; i++) {
    queue.push(i);
  }
  for (int k = 0; k < 1000; k++) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 0; i < 10; i++) {
      queue.push(i);
    }
  }
}

TEST_CASE("PriorityQueue: performance (desc)") {
  std::priority_queue<int> queue;
  for (int i = 1000000; i > 0; i--) {
    queue.push(i);
  }
  for (int k = 1000; k > 0; k--) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 10; i > 0; i--) {
      queue.push(i);
    }
  }
}

TEST_CASE("ContainerPriorityQueue: performance") {
  ContainerQueue<int, std::priority_queue> queue;
  for (int i = 0; i < 1000000; i++) {
    queue.push(i, i);
  }
  for (int k = 0; k < 1000; k++) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 0; i < 10; i++) {
      queue.push(i, i);
    }
  }
}

TEST_CASE("RandomPriorityQueue: performance") {
  ContainerQueue<int, std::priority_queue> queue;
  for (int i = 0; i < 1000000; i++) {
    queue.push_random(i);
  }
  for (int k = 0; k < 1000; k++) {

    for (int i = 0; i < 10; i++) {
      queue.pop();
    }
    for (int i = 0; i < 10; i++) {
      queue.push_random(i);
    }
  }
}