#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/queues.hpp"
#include <chrono>

TEST_CASE("TopKQueue: push") {
  TopKQueue<int, 3> queue;
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
  TopKQueue<int, 3> queue;
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
  TopKQueue<int, 3> queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);

  CHECK(queue.top() == 5);
  queue.push(6);
  CHECK(queue.top() == 6);
}

TEST_CASE("TopKQueue: at") {
  TopKQueue<int, 3> queue;
  queue.push(5);
  queue.push(1);
  queue.push(3);

  CHECK(queue.at(0) == 5);
  CHECK(queue.at(1) == 3);
  CHECK(queue.at(2) == 1);
}

TEST_CASE("TopKQueue: remove_at") {
  TopKQueue<int, 3> queue;
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
  TopKQueue<int, 3> queue;
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
  TopKQueue<int, 3> queue;
  CHECK(queue.size() == 0);

  queue.push(5);
  queue.push(1);
  queue.push(3);
  CHECK(queue.size() == 3);

  queue.push(2);
  queue.push(4);
  CHECK(queue.size() == 5);
}

TEST_CASE("TopKQueue: performance") {
  auto start = std::chrono::high_resolution_clock::now();

  TopKQueue<int, 4> queue;
  for (int i = 0; i < 1000000; i++) {
    queue.push(i);
  }
  for (int i = 0; i < 10000; i++) {
    queue.pop();
  }
  for (int i = 0; i < 10000; i++) {
    queue.push(i);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  MESSAGE("Performance test took ", elapsed.count(), " seconds");
}