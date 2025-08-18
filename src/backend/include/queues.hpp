#pragma once
#include "settings.hpp"
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
#include <type_traits>
#include <utility>
#include <vector>
#include <array>

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


template<typename Q>
concept HasStaticK = requires { { Q::K }  -> std::convertible_to<int>; };

template <typename Q>
concept HasDynamicK = requires(Q q, int i) {
  { q.get_k() } -> std::convertible_to<int>;
  { q.set_k(i) } -> std::convertible_to<void>;
};

template <typename Q>
concept HasTopKInterface = requires(Q q, std::vector<std::size_t>& idx, std::size_t i) {
  typename Q::value_type;
  { q.topk_size() } -> std::convertible_to<std::size_t>;
  { q.get_top_k() };
  { q.remove(idx) };
  { q.remove_at(i) };
  { q.at(i) };
};

template<typename Q>
concept TopKLike = HasTopKInterface<Q> && (HasStaticK<Q> || HasDynamicK<Q>);

template <typename T, int s> class ResizeableArray {
private:
  std::size_t l{};
  std::array<T, s> arr;

public:
  ResizeableArray() = default;
  T &at(std::size_t index) {
    return arr[index];
  }
  T &operator[](int index) {
    return arr[index];
  }
  void remove_at(int index);
  void erase(T *it);
  void insert(T *it, T val);
  void insert_at(int index, T val);
  void push_back(T val) {
    arr[l++] = val;
  }
  void pop_back() {
    l--;
  }
  [[nodiscard]] std::size_t size() const {
    return l;
  }
  void pop_front();
  void push_front(T val);
  [[nodiscard]] const T &front() const {
    return arr[0];
  }
  [[nodiscard]] const T &back() const {
    return arr[l - 1];
  }
  T &front() {
    return arr[0];
  }
  T &back() {
    return arr[l - 1];
  }
  T *begin() {
    return arr.begin();
  }
  T *end() {
    return arr.begin() + l;
  }
  [[nodiscard]] bool empty() const {
    return l == 0;
  }
};

template <typename T> struct Element {
  T value;
  int priority;

  bool operator<(const Element &other) const {
    // if (priority != other.priority) {
    //   return priority < other.priority;
    // }
    // return value < other.value;
    if (priority == other.priority) {
      return value < other.value;
    }
    return priority > other.priority;
  }
};

template <typename T> inline Element<T> make_element(T value, int priority) {
  return Element{value, priority};
}

template <typename T> inline Element<T> make_element(T value) {
  return Element{value, 0};
}

// TODO(wlr): Add support for emplace instead of push to avoid making temporary
// Elements in ContainerQueue

template <typename T, template <typename...> class Queue = std::priority_queue,
          typename Compare = std::less<T>>
class ContainerQueue {
private:
  struct ElementCompare {
    Compare value_compare;
    bool operator()(const Element<T> &lhs, const Element<T> &rhs) const {
      // if(lhs.priority != rhs.priority) {
      //   return lhs.priority < rhs.priority;
      // }
      // return value_compare(lhs.value, rhs.value); 
      if (lhs.priority == rhs.priority) {
        return value_compare(lhs.value, rhs.value);
      }
      return value_compare(rhs.priority, lhs.priority);
    }
  };

  using QueueType = Queue<Element<T>, std::vector<Element<T>>, ElementCompare>;
  static_assert(PriorityQueueConcept<QueueType>, "Queue must satisfy PriorityQueueConcept");

  unsigned long seed = 0;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist;
  QueueType pq;

public:
  using value_type = T;
  using element_type = Element<T>;
  using value_compare = Compare;
  using element_compare = ElementCompare;

  // Conditionally define ContainerQueue::K if QueueType::K exists to satify
  // is_top_k_queue if QueueType is a TopKQueue
  template <typename QT = QueueType>
    requires HasStaticK<QT>
  static constexpr int K = QT::K;

  ContainerQueue() : dist(MIN_PRIORITY, MAX_PRIORITY) {
  }
  ContainerQueue(unsigned long seed) : gen(seed), dist(MIN_PRIORITY, MAX_PRIORITY) {
  }
  ContainerQueue(int min, int max) : gen(0), dist(min, max) {
  }
  ContainerQueue(unsigned long seed, int min, int max) : gen(seed), dist(min, max) {
  }

  void push(T value) {
    pq.push(make_element(value));
  }
  void push(T value, priority_t priority) {
    pq.push(make_element(value, priority));
  }
  void push(Element<T> element) {
    pq.push(element);
  }
  void push_random(T value) {
    pq.push(make_element(value, dist(gen)));
  }
  [[nodiscard]] const T &top() const {
    return pq.top().value;
  }
  const T &top() {
    return const_cast<T &>(std::as_const(*this).top());
  }
  [[nodiscard]] const Element<T> &top_element() const {
    return pq.top();
  };
  const Element<T> &top_element() {
    return const_cast<Element<T> &>(std::as_const(*this).top_element());
  };
  void pop() {
    pq.pop();
  }
  [[nodiscard]] bool empty() const {
    return pq.empty();
  }
  [[nodiscard]] std::size_t size() const {
    return pq.size();
  }
  static consteval int get_k();
  int k() const; 
  int topk_size();
  static consteval bool is_top_k() {
    return TopKLike<QueueType>;
  }
  T &at(std::size_t i);
  void remove_at(std::size_t i);
  void remove(std::vector<std::size_t> &indices);
  auto get_top_k_elements();
  std::vector<T> get_top_k();
  void set_k(int k);
};

static_assert(QueueConcept<ContainerQueue<int, std::priority_queue>>,
              "Queue must satisfy QueueConcept");


template <typename T, int k = 3, typename Container = std::vector<T>,
          typename Compare = std::less<T>>
class TopKQueue {
private:
  // ResizeableArray<T, k> top_k;
  std::vector<T> top_k;
  std::priority_queue<T, Container, Compare> remaining_min_heap;
  Compare cmp;
  std::function<bool(const T &, const T &)> r_cmp;

  void insert_top_k(const T &val);
  void push_front(const T &val) {
    top_k.insert(top_k.begin(), val);
  }
  void pop_front() {
    top_k.erase(top_k.begin());
  }

public:
  using value_type = T;
  using value_compare = Compare;
  static constexpr int K = k;

  TopKQueue() : r_cmp([this](const T &a, const T &b) { return cmp(b, a); }) {
  }

  [[nodiscard]] value_compare value_comp() const {
    return Compare{};
  }
  void push(const T &val);
  void pop();
  [[nodiscard]] const T &top() const;
  T &top() {
    return const_cast<T &>(std::as_const(*this).top());
  }
  T &at(std::size_t i);
  void remove_at(std::size_t i);
  [[nodiscard]] bool empty() const {
    return top_k.empty();
  }
  [[nodiscard]] std::size_t size() const {
    return top_k.size() + remaining_min_heap.size();
  }
  auto &get_top_k() {
    return top_k;
  }
  std::size_t topk_size() {
    return top_k.size();
  }
  static consteval int get_k() {
    return K;
  }
  static consteval bool is_top_k() {
    return true;
  }

  void remove(std::vector<std::size_t> &indices);
};

static_assert(QueueConcept<TopKQueue<int, 3>>, "Queue must satisfy QueueConcept");

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
class DynamicTopKQueue {
  private: 
    std::vector<T> top_k;
    std::priority_queue<T, Container, Compare> remaining_min_heap;
    Compare cmp{};
    std::function<bool(const T &, const T &)> r_cmp;
    int K_ = 1;

    void insert_top_k(const T &val) {
      auto it = std::lower_bound(top_k.begin(), top_k.end(), val, r_cmp);
      top_k.insert(it, val);
    }

    void push_front(const T &val) {
      top_k.insert(top_k.begin(), val);
    }

    void pop_front() {
      top_k.erase(top_k.begin());
    }

    void rebalance() {
      while(static_cast<int>(top_k.size()) > K_) {
        remaining_min_heap.push(top_k.back());
        top_k.pop_back();
      }
      while(static_cast<int>(top_k.size()) < K_ && !remaining_min_heap.empty()) {
        T v = remaining_min_heap.top();
        remaining_min_heap.pop();
        insert_top_k(v);
      }
    }

  public:
    using value_type = T;
    using value_compare = Compare;
    using topk_tag = void;

    DynamicTopKQueue() : r_cmp([this](const T& a, const T& b) { return cmp(b, a); }) {}

    explicit DynamicTopKQueue(int k, Compare c = Compare{}) : cmp(std::move(c)),
        r_cmp([this](const T& a, const T& b) { return cmp(b, a); }),
        K_(std::max(1, k)) {}

    [[nodiscard]] value_compare value_comp() const { return cmp; }

    void push(const T& val) {
      if (static_cast<int>(top_k.size()) < K_) {
        insert_top_k(val);
      } else {
        if (cmp(val, top_k.back())) { // worse than window's worst
          remaining_min_heap.push(val);
        } else {
          remaining_min_heap.push(top_k.back());
          top_k.pop_back();
          insert_top_k(val);
        }
      }
    }

    void pop() {
      assert(!top_k.empty() && "pop() called on an empty DynamicTopKQueue");
      pop_front();
      if (!remaining_min_heap.empty()) {
        T v = remaining_min_heap.top();
        remaining_min_heap.pop();
        insert_top_k(v);
      }
    }

    [[nodiscard]] const T& top() const {
      assert(!top_k.empty() && "top() called on an empty DynamicTopKQueue");
      return top_k.front();
    }
    T& top() { return const_cast<T&>(std::as_const(*this).top()); }

    T& at(std::size_t i) {
      assert(i < top_k.size() && "at() index out of range");
      return top_k[i];
    }

    void remove_at(std::size_t i) {
      assert(i < top_k.size() && "remove_at() index out of range");
      top_k.erase(top_k.begin() + i);
      if (!remaining_min_heap.empty()) {
        T v = remaining_min_heap.top();
        remaining_min_heap.pop();
        insert_top_k(v);
      }
    }

    void remove(std::vector<std::size_t>& indices) {
      std::sort(indices.begin(), indices.end(), std::greater<>());
      for (auto i : indices) remove_at(i);
  }

  [[nodiscard]] bool empty() const { return top_k.empty(); }
  [[nodiscard]] std::size_t size() const { return top_k.size() + remaining_min_heap.size(); }
  auto& get_top_k() { return top_k; }
  std::size_t topk_size() { return top_k.size(); }

  int get_k() const { return K_; }
  void set_k(int k) { 
    K_ = std::max(1, k); rebalance(); }
};


template <int k> struct TopKQueueHelper {
  template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
  using queue_type = TopKQueue<T, k, Container, Compare>;
};

static_assert(QueueConcept<DynamicTopKQueue<int>>, "DynamicTopKQueue must satisfy QueueConcept");
static_assert(TopKLike<DynamicTopKQueue<int>>, "DynamicTopKQueue must satisfy TopKLike");
static_assert(QueueConcept<ContainerQueue<int, DynamicTopKQueue>>,              "ContainerQueue of DynamicTopKQueue must satisfy QueueConcept");

static_assert(TopKLike<TopKQueue<int, 3>>, "TopKQueue must satisfy is_topk_queue");

static_assert(TopKLike<TopKQueueHelper<3>::queue_type<int>>);

static_assert(QueueConcept<ContainerQueue<int, TopKQueueHelper<3>::queue_type>>,
              "ContainerQueue of TopKQueue must satisfy QueueConcept");
static_assert(QueueConcept<ContainerQueue<int, DynamicTopKQueue>>,
              "ContainerQueue of DynamicTopKQueue must satisfy QueueConcept");

using Top3Queue = TopKQueue<int, 3>;
using Top10Queue = TopKQueue<int, 10>;

template <typename T, int S> void print(TopKQueue<T, S> &q);
template <QueueConcept Q> std::vector<typename Q::value_type> as_vector(Q &q);
template <WrappedQueueConcept Q> std::vector<typename Q::element_type> as_vector(Q &q);
template <WrappedQueueConcept Q> void print_table(Q &q);

template <typename T, int s> void ResizeableArray<T, s>::remove_at(int index) {
  for (int i = index; i < l - 1; i++) {
    assert(i + 1 < s);
    arr[i] = arr[i + 1];
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::erase(T *it) {
  for (T *i = it; i != arr.begin() + l - 1; i++) {
    // check in bounds
    assert(i + 1 < arr.end());
    *i = *(i + 1);
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::insert(T *it, T val) {
  for (T *i = arr.begin() + l - 1; i >= it; i--) {
    // check in bounds
    assert(i + 1 < arr.end());
    *(i + 1) = *i;
  }
  *it = val;
  l++;
}

template <typename T, int s> void ResizeableArray<T, s>::insert_at(int index, T val) {
  for (int i = l - 1; i >= index; i--) {
    assert(i + 1 < s);
    arr[i + 1] = arr[i];
  }
  arr[index] = val;
  l++;
}

template <typename T, int s> void ResizeableArray<T, s>::pop_front() {
  for (int i = 0; i < l - 1; i++) {
    assert(i + 1 < s);
    arr[i] = arr[i + 1];
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::push_front(T val) {
  for (int i = l - 1; i >= 0; i--) {
    assert(i + 1 < s);
    arr[i + 1] = arr[i];
  }
  arr[0] = val;
  l++;
}

template <typename T, template <typename...> class Queue, typename Compare>
 consteval int ContainerQueue<T, Queue, Compare>::get_k() {
  if constexpr (HasStaticK<QueueType>) {
     return QueueType::K;
   } else {
     return 1;
   }
 }

template <typename T, template <typename...> class Queue, typename Compare>
int ContainerQueue<T, Queue, Compare>::k() const {
  if constexpr (HasStaticK<QueueType>) {
    return QueueType::K;
  } else if constexpr (HasTopKInterface<QueueType>) {
    // pq may have only non-const get_k()
    return const_cast<QueueType&>(pq).get_k();
  } else {
    return 1;
  }
}

template <typename T, template <typename...> class Queue, typename Compare>
void ContainerQueue<T, Queue, Compare>::remove(std::vector<std::size_t> &indices) {
  if constexpr (TopKLike<QueueType>) {
    pq.remove(indices);
  } else {
    throw std::out_of_range("remove() called on a non-top-k queue");
  }
}

template <typename T, template <typename...> class Queue, typename Compare>
int ContainerQueue<T, Queue, Compare>::topk_size() {
  if constexpr (TopKLike<QueueType>) {
    return static_cast<int>(pq.topk_size());
  } else {
    return 1;
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
T &ContainerQueue<T, Queue, Compare>::at(std::size_t i) {
  if constexpr (TopKLike<QueueType>) {
    return pq.at(i);
  } else {
    throw std::out_of_range("at() called on a non-top-k queue");
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
void ContainerQueue<T, Queue, Compare>::remove_at(std::size_t i) {
  if constexpr (TopKLike<QueueType>) {
    pq.remove_at(i);
  } else {
    throw std::out_of_range("remove_at() called on a non-top-k queue");
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
auto ContainerQueue<T, Queue, Compare>::get_top_k_elements() {
  if constexpr (TopKLike<QueueType>) {
    return pq.get_top_k();
  } else {
    return std::vector<typename ContainerQueue::element_type>{this->top_element()};
  }
}

template <typename T, template <typename...> class Queue, typename Compare>
std::vector<T> ContainerQueue<T, Queue, Compare>::get_top_k() {
  std::vector<T> top_k_values;
  if constexpr (TopKLike<QueueType>) {
    auto &top_k = pq.get_top_k();
    for (auto it = top_k.begin(); it != top_k.end(); ++it) {
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*it)>, Element<T>>) {
        top_k_values.push_back(it->value);
      } else {
        top_k_values.push_back(*it);
      }
    }
   } else {
     return std::vector<T>{this->top()};
   }
   return top_k_values;
 }

template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::insert_top_k(const T &val) {
  auto it = std::lower_bound(top_k.begin(), top_k.end(), val, r_cmp);
  top_k.insert(it, val);
}

template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::push(const T &val) {
  if (top_k.size() < k) {
    insert_top_k(val);
  } else {
    if (cmp(val, top_k.back())) {
      remaining_min_heap.push(val);
    } else {
      remaining_min_heap.push(top_k.back());
      top_k.pop_back();
      insert_top_k(val);
    }
  }
}

template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::pop() {
  assert(!top_k.empty() && "pop() called on an empty queue");
  pop_front();

  if (!remaining_min_heap.empty()) {
    top_k.push_back(remaining_min_heap.top());
    remaining_min_heap.pop();
    // T v = remaining_min_heap.top();
    // remaining_min_heap.pop();
    // insert_top_k(v);
  }
}

template <typename T, int k, typename Container, typename Compare>
const T &TopKQueue<T, k, Container, Compare>::top() const {
  assert(!top_k.empty() && "top() called on an empty queue");
  return top_k.front();
}

template <typename T, int k, typename Container, typename Compare>
T &TopKQueue<T, k, Container, Compare>::at(std::size_t i) {
  assert(i < top_k.size() && "at() called with an index out of range");
  return top_k[i];
}
template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::remove_at(std::size_t i) {
  assert(i < top_k.size() && "remove_at() called with an index out of range");
  top_k.erase(top_k.begin() + i);

  if (!remaining_min_heap.empty()) {
    top_k.push_back(remaining_min_heap.top());
    remaining_min_heap.pop();
    // T v = remaining_min_heap.top();
    // remaining_min_heap.pop();
    // insert_top_k(v);
  }
}

template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::remove(std::vector<std::size_t> &indices) {
  std::sort(indices.begin(), indices.end(), std::greater<>());
  for (auto i : indices) {
    remove_at(i);
  }
}


template <typename T, int S> void print(TopKQueue<T, S> &q) {
  auto &top_k = q.get_top_k();
  for (auto it = top_k.begin(); it != top_k.end(); ++it) {
    std::cout << *it << " ";
  }
}

template <QueueConcept Q> std::vector<typename Q::value_type> as_vector(Q &q) {
  typedef typename Q::value_type Value_t;
  // drain store and refill queue
  std::vector<Value_t> elements;
  while (!q.empty()) {
    elements.push_back(q.top());
    q.pop();
  }
  for (auto &element : elements) {
    q.push(element);
  }
  return elements;
}

template <WrappedQueueConcept Q> std::vector<typename Q::element_type> as_vector(Q &q) {
  typedef typename Q::element_type Element_t;
  // drain store and refill queue
  std::vector<Element_t> elements;
  while (!q.empty()) {
    elements.push_back(q.top_element());
    q.pop();
  }
  for (auto &element : elements) {
    q.push(element);
  }
  return elements;
}

template <typename T, template <typename...> class Queue, typename Compare>
void ContainerQueue<T, Queue, Compare>::set_k(int k) {
  if constexpr (HasDynamicK<QueueType>) {
    pq.set_k(k);
  } else {
    // No-op for non-dynamic queues
    (void)k;
  }
}