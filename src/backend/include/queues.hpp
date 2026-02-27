#pragma once
#include "settings.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <queue>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

constexpr int MIN_PRIORITY = 0;
constexpr int MAX_PRIORITY = 100;

template <typename Q>
concept QueueConcept = requires(Q q) {
  typename Q::value_type;
  { q.push(std::declval<typename Q::value_type>()) };
  { q.pop() };
  { q.top() } -> std::convertible_to<typename Q::value_type>;
  { q.size() } -> std::convertible_to<std::size_t>;
  { q.empty() } -> std::convertible_to<bool>;
};

template <typename Q>
concept PriorityQueueConcept = QueueConcept<Q> && requires(Q q) { typename Q::value_compare; };

template <typename Q>
concept WrappedQueueConcept = QueueConcept<Q> && requires(Q q) {
  typename Q::element_type;
  typename Q::element_compare;
};

template <typename Q>
concept HasStaticK = requires {
  { Q::K } -> std::convertible_to<int>;
};

template <typename Q>
concept HasDynamicK = requires(Q q, int i) {
  { q.get_k() } -> std::convertible_to<int>;
  { q.set_k(i) } -> std::convertible_to<void>;
};

template <typename Q>
concept HasTopKInterface = requires(Q q, std::vector<std::size_t> &idx, std::size_t i) {
  typename Q::value_type;
  { q.topk_size() } -> std::convertible_to<std::size_t>;
  { q.get_top_k() };
  { q.remove(idx) };
  { q.remove_at(i) };
  { q.at(i) };
};

template <typename Q>
concept TopKLike = HasTopKInterface<Q> && (HasStaticK<Q> || HasDynamicK<Q>);

template <typename T, int s> class ResizeableArray {
private:
  std::size_t l{};
  std::array<T, s> arr;

public:
  ResizeableArray() = default;

  T &at(std::size_t index) { return arr[index]; }
  T &operator[](int index) { return arr[index]; }

  void remove_at(int index) {
    assert(index >= 0 && index < static_cast<int>(l));
    std::move(arr.begin() + index + 1, arr.begin() + l, arr.begin() + index);
    l--;
  }

  void erase(T *it) {
    assert(it >= arr.data() && it < arr.data() + l);
    std::move(it + 1, arr.data() + l, it);
    l--;
  }

  void insert(T *it, T val) {
    assert(l < s);
    std::move_backward(it, arr.data() + l, arr.data() + l + 1);
    *it = std::move(val);
    l++;
  }

  void insert_at(int index, T val) {
    assert(l < s);
    assert(index >= 0 && index <= static_cast<int>(l));
    std::move_backward(arr.begin() + index, arr.begin() + l, arr.begin() + l + 1);
    arr[index] = std::move(val);
    l++;
  }

  void push_back(T val) {
    assert(l < s);
    arr[l++] = std::move(val);
  }

  void pop_back() {
    assert(l > 0);
    l--;
  }

  [[nodiscard]] std::size_t size() const noexcept { return l; }

  void pop_front() {
    assert(l > 0);
    std::move(arr.begin() + 1, arr.begin() + l, arr.begin());
    l--;
  }

  void push_front(T val) {
    assert(l < s);
    std::move_backward(arr.begin(), arr.begin() + l, arr.begin() + l + 1);
    arr[0] = std::move(val);
    l++;
  }

  [[nodiscard]] const T &front() const { return arr[0]; }
  [[nodiscard]] const T &back() const { return arr[l - 1]; }
  T &front() { return arr[0]; }
  T &back() { return arr[l - 1]; }
  T *begin() { return arr.data(); }
  T *end() { return arr.data() + l; }
  const T *begin() const { return arr.data(); }
  const T *end() const { return arr.data() + l; }

  [[nodiscard]] bool empty() const noexcept { return l == 0; }
};

template <typename T> struct Element {
  T value;
  int priority;

  Element() = default;
  Element(T v, int p) : value(std::move(v)), priority(p) {}
  explicit Element(T v) : value(std::move(v)), priority(0) {}

  bool operator<(const Element &other) const {
    if (priority == other.priority) {
      return value > other.value;
    }
    return priority > other.priority;
  }
};

template <typename T> struct PackedElement {
  static_assert(sizeof(T) <= 4, "T must be 32-bit or smaller to pack");

  uint64_t key{};

  PackedElement() = default;
  PackedElement(T v, int p) {
    uint32_t u_val;
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      u_val = static_cast<uint32_t>(v) ^ 0x80000000u;
    } else if constexpr (std::is_same_v<T, float>) {
      u_val = std::bit_cast<uint32_t>(v);
      if (u_val & 0x80000000u) u_val = ~u_val;
      else u_val |= 0x80000000u;
    } else {
      u_val = static_cast<uint32_t>(v);
    }
    key = (static_cast<uint64_t>(static_cast<uint32_t>(p)) << 32) | u_val;
  }

  [[nodiscard]] T value() const {
    uint32_t u_val = static_cast<uint32_t>(key & 0xFFFFFFFFu);
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      return static_cast<T>(u_val ^ 0x80000000u);
    } else if constexpr (std::is_same_v<T, float>) {
      if (u_val & 0x80000000u) u_val &= ~0x80000000u;
      else u_val = ~u_val;
      return std::bit_cast<float>(u_val);
    } else {
      return static_cast<T>(u_val);
    }
  }

  [[nodiscard]] int priority() const {
    return static_cast<int>(static_cast<uint32_t>(key >> 32));
  }

  bool operator<(const PackedElement &other) const { return key < other.key; }
  bool operator==(const PackedElement &other) const { return key == other.key; }
};

template <typename T>
using OptimalElement =
    std::conditional_t<(sizeof(T) <= 4), PackedElement<T>, Element<T>>;

template <typename T> [[nodiscard]] inline const T &element_value(const Element<T> &e) {
  return e.value;
}
template <typename T> [[nodiscard]] inline T element_value(const PackedElement<T> &e) {
  return e.value();
}

template <typename T> [[nodiscard]] inline int element_priority(const Element<T> &e) {
  return e.priority;
}
template <typename T> [[nodiscard]] inline int element_priority(const PackedElement<T> &e) {
  return e.priority();
}

template <typename T> inline OptimalElement<T> make_element(T value, int priority) {
  return OptimalElement<T>{std::move(value), priority};
}

template <typename T> inline OptimalElement<T> make_element(T value) {
  return OptimalElement<T>{std::move(value), 0};
}

template <typename T, template <typename...> class Queue = std::priority_queue,
          typename Compare = std::less<T>>
class ContainerQueue {
private:
  using StoredElement = OptimalElement<T>;

  struct ElementCompare {
    [[no_unique_address]] Compare value_compare;
    bool operator()(const StoredElement &lhs, const StoredElement &rhs) const {
      const auto lhs_val = element_value(lhs);
      const auto rhs_val = element_value(rhs);
      if (value_compare(rhs_val, lhs_val)) return true;
      if (value_compare(lhs_val, rhs_val)) return false;
      return element_priority(lhs) > element_priority(rhs);
    }
  };

  using QueueType = Queue<StoredElement, std::vector<StoredElement>, ElementCompare>;
  static_assert(PriorityQueueConcept<QueueType>, "Queue must satisfy PriorityQueueConcept");

  unsigned long seed = 0;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist;
  QueueType pq;

public:
  using value_type = T;
  using element_type = StoredElement;
  using value_compare = Compare;
  using element_compare = ElementCompare;

  template <typename QT = QueueType> requires HasStaticK<QT>
  static constexpr int K = QT::K;

  ContainerQueue() : dist(MIN_PRIORITY, MAX_PRIORITY) {}
  explicit ContainerQueue(unsigned long seed) : gen(seed), dist(MIN_PRIORITY, MAX_PRIORITY) {}
  ContainerQueue(int min, int max) : gen(0), dist(min, max) {}
  ContainerQueue(unsigned long seed, int min, int max) : gen(seed), dist(min, max) {}

  template <typename... Args> void emplace(Args &&...args) {
    if constexpr (requires(QueueType &q) { q.emplace(std::forward<Args>(args)...); }) {
      pq.emplace(std::forward<Args>(args)...);
    } else {
      pq.push(element_type{std::forward<Args>(args)...});
    }
  }

  void push(T value) { emplace(std::move(value), 0); }

  void push(T value, priority_t priority) {
    emplace(std::move(value), static_cast<int>(priority));
  }

  void push(element_type element) { pq.push(std::move(element)); }
  void push_random(T value) { emplace(std::move(value), dist(gen)); }

  [[nodiscard]] T top() const { return element_value(pq.top()); }
  [[nodiscard]] const element_type &top_element() const { return pq.top(); }
  element_type &top_element() { return const_cast<element_type &>(std::as_const(*this).top_element()); }

  void pop() { pq.pop(); }
  
  [[nodiscard]] bool empty() const noexcept { return pq.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return pq.size(); }

  static consteval int get_k() {
    if constexpr (HasStaticK<QueueType>) return QueueType::K;
    else return 1;
  }

  int k() const {
    if constexpr (HasStaticK<QueueType>) return QueueType::K;
    else if constexpr (HasTopKInterface<QueueType>) return const_cast<QueueType &>(pq).get_k();
    else return 1;
  }

  int topk_size() {
    if constexpr (TopKLike<QueueType>) return static_cast<int>(pq.topk_size());
    else return 1;
  }

  static consteval bool is_top_k() { return TopKLike<QueueType>; }

  T &at(std::size_t i) {
    if constexpr (TopKLike<QueueType>) return pq.at(i);
    else throw std::out_of_range("at() called on a non-top-k queue");
  }

  void remove_at(std::size_t i) {
    if constexpr (TopKLike<QueueType>) pq.remove_at(i);
    else throw std::out_of_range("remove_at() called on a non-top-k queue");
  }

  void remove(std::vector<std::size_t> &indices) {
    if constexpr (TopKLike<QueueType>) pq.remove(indices);
    else throw std::out_of_range("remove() called on a non-top-k queue");
  }

  auto get_top_k_elements() {
    if constexpr (TopKLike<QueueType>) return pq.get_top_k();
    else return std::vector<element_type>{this->top_element()};
  }

  std::vector<T> get_top_k() {
    std::vector<T> top_k_values;
    if constexpr (TopKLike<QueueType>) {
      auto &top_k_ref = pq.get_top_k();
      top_k_values.reserve(top_k_ref.size());
      for (const auto &elem : top_k_ref) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(elem)>, element_type>) {
          top_k_values.push_back(element_value(elem));
        } else top_k_values.push_back(elem);
      }
    } else {
      top_k_values.push_back(this->top());
    }
    return top_k_values;
  }

  void set_k(int k_val) {
    if constexpr (HasDynamicK<QueueType>) pq.set_k(k_val);
    else (void)k_val;
  }
};

static_assert(QueueConcept<ContainerQueue<int, std::priority_queue>>, "Queue must satisfy QueueConcept");

template <typename T, int k = 3, typename Container = std::vector<T>,
          typename Compare = std::less<T>>
class TopKQueue {
private:
  std::vector<T> top_k;
  std::priority_queue<T, Container, Compare> remaining_min_heap;
  [[no_unique_address]] Compare cmp{};

  void insert_top_k(T val) {
    auto it = std::lower_bound(top_k.begin(), top_k.end(), val, 
                               [this](const T &a, const T &b) { return cmp(b, a); });
    top_k.insert(it, std::move(val));
  }

public:
  using value_type = T;
  using value_compare = Compare;
  static constexpr int K = k;

  TopKQueue() = default;

  [[nodiscard]] value_compare value_comp() const { return cmp; }

  void push(T val) {
    if (top_k.size() < k) {
      insert_top_k(std::move(val));
    } else {
      if (cmp(val, top_k.back())) {
        remaining_min_heap.push(std::move(val));
      } else {
        remaining_min_heap.push(std::move(top_k.back()));
        top_k.pop_back();
        insert_top_k(std::move(val));
      }
    }
  }

  void pop() {
    assert(!top_k.empty() && "pop() called on an empty queue");
    top_k.erase(top_k.begin());

    if (!remaining_min_heap.empty()) {
      T v = remaining_min_heap.top();
      remaining_min_heap.pop();
      top_k.push_back(std::move(v));
    }
  }

  [[nodiscard]] const T &top() const {
    assert(!top_k.empty() && "top() called on an empty queue");
    return top_k.front();
  }
  
  T &top() { return const_cast<T &>(std::as_const(*this).top()); }

  T &at(std::size_t i) {
    assert(i < top_k.size() && "at() index out of range");
    return top_k[i];
  }

  void remove_at(std::size_t i) {
    assert(i < top_k.size() && "remove_at() index out of range");
    top_k.erase(top_k.begin() + i);

    if (!remaining_min_heap.empty()) {
      T v = remaining_min_heap.top();
      remaining_min_heap.pop();
      top_k.push_back(std::move(v));
    }
  }

  void remove(std::vector<std::size_t> &indices) {
    std::sort(indices.begin(), indices.end(), std::greater<>());
    for (auto i : indices) {
      remove_at(i);
    }
  }

  [[nodiscard]] bool empty() const noexcept { return top_k.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return top_k.size() + remaining_min_heap.size(); }
  
  auto &get_top_k() { return top_k; }
  const auto &get_top_k() const { return top_k; }
  
  std::size_t topk_size() const noexcept { return top_k.size(); }
  static consteval int get_k() { return K; }
  static consteval bool is_top_k() { return true; }
};

static_assert(QueueConcept<TopKQueue<int, 3>>, "Queue must satisfy QueueConcept");

template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
class DynamicTopKQueue {
private:
  std::vector<T> top_k;
  std::priority_queue<T, Container, Compare> remaining_min_heap;
  [[no_unique_address]] Compare cmp{};
  int K_ = 1;

  void insert_top_k(T val) {
    auto it = std::lower_bound(top_k.begin(), top_k.end(), val, 
                               [this](const T &a, const T &b) { return cmp(b, a); });
    top_k.insert(it, std::move(val));
  }

  void rebalance() {
    while (static_cast<int>(top_k.size()) > K_) {
      remaining_min_heap.push(std::move(top_k.back()));
      top_k.pop_back();
    }
    while (static_cast<int>(top_k.size()) < K_ && !remaining_min_heap.empty()) {
      T v = remaining_min_heap.top();
      remaining_min_heap.pop();
      top_k.push_back(std::move(v));
    }
  }

public:
  using value_type = T;
  using value_compare = Compare;
  using topk_tag = void;

  DynamicTopKQueue() = default;

  explicit DynamicTopKQueue(int k, Compare c = Compare{})
      : cmp(std::move(c)), K_(std::max(1, k)) {}

  [[nodiscard]] value_compare value_comp() const { return cmp; }

  void push(T val) {
    if (static_cast<int>(top_k.size()) < K_) {
      insert_top_k(std::move(val));
    } else {
      if (cmp(val, top_k.back())) {
        remaining_min_heap.push(std::move(val));
      } else {
        remaining_min_heap.push(std::move(top_k.back()));
        top_k.pop_back();
        insert_top_k(std::move(val));
      }
    }
  }

  void pop() {
    assert(!top_k.empty() && "pop() called on an empty DynamicTopKQueue");
    top_k.erase(top_k.begin());
    if (!remaining_min_heap.empty()) {
      T v = remaining_min_heap.top();
      remaining_min_heap.pop();
      top_k.push_back(std::move(v));
    }
  }

  [[nodiscard]] const T &top() const {
    assert(!top_k.empty() && "top() called on an empty DynamicTopKQueue");
    return top_k.front();
  }
  
  T &top() { return const_cast<T &>(std::as_const(*this).top()); }

  T &at(std::size_t i) {
    assert(i < top_k.size() && "at() index out of range");
    return top_k[i];
  }

  void remove_at(std::size_t i) {
    assert(i < top_k.size() && "remove_at() index out of range");
    top_k.erase(top_k.begin() + i);
    if (!remaining_min_heap.empty()) {
      T v = remaining_min_heap.top();
      remaining_min_heap.pop();
      top_k.push_back(std::move(v));
    }
  }

  void remove(std::vector<std::size_t> &indices) {
    std::sort(indices.begin(), indices.end(), std::greater<>());
    for (auto i : indices) {
      remove_at(i);
    }
  }

  [[nodiscard]] bool empty() const noexcept { return top_k.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return top_k.size() + remaining_min_heap.size(); }
  
  auto &get_top_k() { return top_k; }
  const auto &get_top_k() const { return top_k; }
  
  std::size_t topk_size() const noexcept { return top_k.size(); }
  
  int get_k() const noexcept { return K_; }
  void set_k(int k) {
    K_ = std::max(1, k);
    rebalance();
  }
};

template <int k> struct TopKQueueHelper {
  template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
  using queue_type = TopKQueue<T, k, Container, Compare>;
};

static_assert(QueueConcept<DynamicTopKQueue<int>>, "DynamicTopKQueue must satisfy QueueConcept");
static_assert(TopKLike<DynamicTopKQueue<int>>, "DynamicTopKQueue must satisfy TopKLike");
static_assert(QueueConcept<ContainerQueue<int, DynamicTopKQueue>>,
              "ContainerQueue of DynamicTopKQueue must satisfy QueueConcept");

static_assert(TopKLike<TopKQueue<int, 3>>, "TopKQueue must satisfy is_topk_queue");
static_assert(TopKLike<TopKQueueHelper<3>::queue_type<int>>);

static_assert(QueueConcept<ContainerQueue<int, TopKQueueHelper<3>::queue_type>>,
              "ContainerQueue of TopKQueue must satisfy QueueConcept");

using Top3Queue = TopKQueue<int, 3>;
using Top10Queue = TopKQueue<int, 10>;

template <WrappedQueueConcept Q> void print_table(Q &q);

template <typename T, int S> void print(TopKQueue<T, S> &q) {
  for (const auto &elem : q.get_top_k()) {
    std::cout << elem << " ";
  }
}

template <QueueConcept Q> std::vector<typename Q::value_type> as_vector(Q &q) {
  using Value_t = typename Q::value_type;
  std::vector<Value_t> elements;
  elements.reserve(q.size()); // Pre-allocating to avoid multiple resize passes
  while (!q.empty()) {
    elements.push_back(q.top());
    q.pop();
  }
  for (const auto &element : elements) {
    q.push(element);
  }
  return elements;
}

template <WrappedQueueConcept Q> std::vector<typename Q::element_type> as_vector(Q &q) {
  using Element_t = typename Q::element_type;
  std::vector<Element_t> elements;
  elements.reserve(q.size());
  while (!q.empty()) {
    elements.push_back(q.top_element());
    q.pop();
  }
  for (const auto &element : elements) {
    q.push(element);
  }
  return elements;
}
