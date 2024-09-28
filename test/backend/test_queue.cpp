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