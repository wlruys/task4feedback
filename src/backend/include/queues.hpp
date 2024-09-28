#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <queue>
#include <random>
#include <stdexcept>
#include <tabulate/table.hpp>
#include <tabulate/tabulate.hpp>
#include <type_traits>
#include <utility>
#include <vector>

constexpr int MIN_PRIORITY = 0;
constexpr int MAX_PRIORITY = 100;

template <typename Q>
concept QueueConcept = requires(Q q) {
  typename Q::value_type;
  {q.push(std::declval<typename Q::value_type>())};
  {q.pop()};
  { q.top() } -> std::convertible_to<typename Q::value_type>;
  { q.size() } -> std::convertible_to<std::size_t>;
  { q.empty() } -> std::convertible_to<bool>;
};

template <typename Q>
concept PriorityQueueConcept = QueueConcept<Q> && requires(Q q) {
  typename Q::value_compare;
};

template <typename Q>
concept WrappedQueueConcept = QueueConcept<Q> && requires(Q q) {
  typename Q::element_type;
  typename Q::element_compare;
};

template <typename T>
concept is_top_k_queue = requires {
  { T::k } -> std::convertible_to<int>;
};

template <typename T, int s> class ResizeableArray {
private:
  std::size_t l{};
  std::array<T, s> arr;

public:
  ResizeableArray() = default;
  T &at(int index) { return arr[index]; }
  T &operator[](int index) { return arr[index]; }
  void remove_at(int index);
  void erase(T *it);
  void insert(T *it, T val);
  void insert_at(int index, T val);
  void push_back(T val) { arr[l++] = val; }
  void pop_back() { l--; }
  [[nodiscard]] std::size_t size() const { return l; }
  void pop_front();
  void push_front(T val);
  [[nodiscard]] const T &front() const { return arr[0]; }
  [[nodiscard]] const T &back() const { return arr[l - 1]; }
  T &front() { return arr[0]; }
  T &back() { return arr[l - 1]; }
  T *begin() { return arr.begin(); }
  T *end() { return arr.begin() + l; }
  [[nodiscard]] bool empty() const { return l == 0; }
};

template <typename T, template <typename...> class Queue = std::priority_queue,
          typename Compare = std::less<T>>
class ContainerQueue {
private:
  template <typename U> struct Element {
    U value;
    int priority;
  };

  struct ElementCompare {
    Compare compare;
    bool operator()(const Element<T> &lhs, const Element<T> &rhs) const {
      return compare(lhs.priority, rhs.priority);
    }
  };

  Element<T> make_element(T value, int priority) {
    return Element{value, priority};
  }

  Element<T> make_element(T value) { return Element{value, 0}; }

  using QueueType = Queue<Element<T>, std::vector<Element<T>>, ElementCompare>;
  static_assert(PriorityQueueConcept<QueueType>,
                "Queue must satisfy PriorityQueueConcept");

  unsigned long seed = 0;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist;
  QueueType pq;

public:
  using value_type = T;
  using element_type = Element<T>;
  using value_compare = Compare;
  using element_compare = ElementCompare;

  ContainerQueue() : dist(MIN_PRIORITY, MAX_PRIORITY) {}
  ContainerQueue(unsigned long seed)
      : gen(seed), dist(MIN_PRIORITY, MAX_PRIORITY) {}
  ContainerQueue(int min, int max) : gen(0), dist(min, max) {}
  ContainerQueue(unsigned long seed, int min, int max)
      : gen(seed), dist(min, max) {}

  void push(T value) { pq.push(make_element(value)); }
  void push(T value, int priority) { pq.push(make_element(value, priority)); }
  void push(Element<T> element) { pq.push(element); }
  void push_random(T value) { pq.push(make_element(value, dist(gen))); }
  [[nodiscard]] const T &top() const { return pq.top().value; }
  const T &top() { return const_cast<T &>(std::as_const(*this).top()); }
  [[nodiscard]] const Element<T> &top_element() const { return pq.top(); };
  const Element<T> &top_element() {
    return const_cast<Element<T> &>(std::as_const(*this).top_element());
  };
  void pop() { pq.pop(); }
  [[nodiscard]] bool empty() const { return pq.empty(); }
  [[nodiscard]] std::size_t size() const { return pq.size(); }
  static consteval int get_k();
  int topk_size();
  static consteval bool is_top_k() { return is_top_k_queue<QueueType>; }
  T &at(std::size_t i);
  void remove_at(std::size_t i);
  auto get_top_k_elements();
  std::vector<T> get_top_k();
};

static_assert(QueueConcept<ContainerQueue<int, std::priority_queue>>,
              "Queue must satisfy QueueConcept");

template <typename T, int k = 3, typename Container = std::vector<T>,
          typename Compare = std::less<T>>
class TopKQueue {
private:
  ResizeableArray<T, k> top_k;
  std::priority_queue<T, Container, Compare> remaining_min_heap;
  Compare cmp;
  std::function<bool(const T &, const T &)> r_cmp;

  void insert_top_k(const T &val);
  void push_front(const T &val) { top_k.insert(top_k.begin(), val); }
  void pop_front() { top_k.erase(top_k.begin()); }

public:
  using value_type = T;
  using value_compare = Compare;
  static constexpr int K = k;

  TopKQueue() : r_cmp([this](const T &a, const T &b) { return !cmp(a, b); }) {}

  [[nodiscard]] value_compare value_comp() const { return Compare{}; }
  void push(const T &val);
  void pop();
  [[nodiscard]] const T &top() const;
  T &top() { return const_cast<T &>(std::as_const(*this).top()); }
  T &at(std::size_t i);
  void remove_at(std::size_t i);
  [[nodiscard]] bool empty() const { return top_k.empty(); }
  [[nodiscard]] std::size_t size() const {
    return top_k.size() + remaining_min_heap.size();
  }
  auto &get_top_k() { return top_k; }
  std::size_t topk_size() { return top_k.size(); }
  static consteval int get_k() { return K; }
  static consteval bool is_top_k() { return true; }
};

static_assert(QueueConcept<TopKQueue<int, 3>>,
              "Queue must satisfy QueueConcept");

template <int k> struct TopKQueueHelper {
  template <typename T, typename Container = std::vector<T>,
            typename Compare = std::less<T>>
  using queue_type = TopKQueue<T, k, Container, Compare>;
};

using Top3Queue = TopKQueue<int, 3>;
using Top10Queue = TopKQueue<int, 10>;

template <typename T, int S> void print(TopKQueue<T, S> &q);
template <QueueConcept Q> std::vector<typename Q::value_type> as_vector(Q &q);
template <WrappedQueueConcept Q>
std::vector<typename Q::element_type> as_vector(Q &q);
template <WrappedQueueConcept Q> void print_table(Q &q);
