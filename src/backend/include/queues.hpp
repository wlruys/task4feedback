#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
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

  void remove_at(int index) {
    for (int i = index; i < l - 1; i++) {
      arr[i] = arr[i + 1];
    }
    l--;
  }

  void erase(T *it) {
    for (T *i = it; i != arr.begin() + l - 1; i++) {
      *i = *(i + 1);
    }
    l--;
  }

  void insert(T *it, T val) {
    for (T *i = arr.begin() + l - 1; i >= it; i--) {
      *(i + 1) = *i;
    }
    *it = val;
    l++;
  }

  void insert_at(int index, T val) {
    for (int i = l - 1; i >= index; i--) {
      arr[i + 1] = arr[i];
    }
    arr[index] = val;
    l++;
  }

  void push_back(T val) { arr[l++] = val; }

  void pop_back() { l--; }

  T &operator[](int index) { return arr[index]; }

  [[nodiscard]] std::size_t size() const { return l; }

  void pop_front() {
    for (int i = 0; i < l - 1; i++) {
      arr[i] = arr[i + 1];
    }
    l--;
  }

  void push_front(T val) {
    for (int i = l - 1; i >= 0; i--) {
      arr[i + 1] = arr[i];
    }
    arr[0] = val;
    l++;
  }

  [[nodiscard]] const T &front() const { return arr[0]; }

  T &front() { return arr[0]; }

  T &back() { return arr[l - 1]; }

  T *begin() { return arr.begin(); }

  T *end() { return arr.begin() + l; }

  [[nodiscard]] bool empty() const { return l == 0; }

  void reserve(int new_size) {}
};

template <typename T, template <typename...> class Queue = std::priority_queue,
          typename Compare = std::less<T>>
class Randomizer {
private:
  unsigned long seed = 0;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist;

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

  using QueueType = Queue<Element<T>, std::vector<Element<T>>, ElementCompare>;
  QueueType pq;

  static_assert(PriorityQueueConcept<QueueType>,
                "Queue must satisfy PriorityQueueConcept");

  Element<T> make_element(T value, int priority) {
    return Element{value, priority};
  }

  Element<T> make_element(T value) { return Element{value, 0}; }

public:
  using value_type = T;
  using element_type = Element<T>;
  using value_compare = Compare;
  using element_compare = ElementCompare;

  Randomizer() : dist(MIN_PRIORITY, MAX_PRIORITY) {}
  Randomizer(unsigned long seed)
      : gen(seed), dist(MIN_PRIORITY, MAX_PRIORITY) {}
  Randomizer(int min, int max) : gen(0), dist(min, max) {}
  Randomizer(unsigned long seed, int min, int max)
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

  static consteval int get_k() {
    if constexpr (is_top_k_queue<QueueType>) {
      return QueueType::K;
    } else {
      return 1;
    }
  }

  int topk_size() {
    if constexpr (is_top_k_queue<QueueType>) {
      return pq.topk_size();
    } else {
      return 1;
    }
  }
  static consteval bool is_top_k() { return is_top_k_queue<QueueType>; }

  T &at(std::size_t i) {
    if constexpr (is_top_k_queue<QueueType>) {
      return pq.get_top_k().at(i);
    } else {
      throw std::out_of_range("at() called on a non-top-k queue");
    }
  }

  void remove_at(std::size_t i) {
    if constexpr (is_top_k_queue<QueueType>) {
      pq.get_top_k().remove_at(i);
    } else {
      throw std::out_of_range("remove_at() called on a non-top-k queue");
    }
  }

  auto &get_top_k_elements() {
    if constexpr (is_top_k_queue<QueueType>) {
      return pq.get_top_k();
    } else {
      return std::vector<T>{this->top()};
    }
  }

  auto get_top_k() {
    auto &top_k = pq.get_top_k();

    std::vector<T> top_k_values;
    for (auto it = top_k.begin(); it != top_k.end(); ++it) {
      top_k_values.push_back(it->value);
    }
    return top_k_values;
  }
};

static_assert(QueueConcept<Randomizer<int, std::priority_queue>>,
              "Queue must satisfy QueueConcept");

template <typename T, int k = 3, typename Container = std::vector<T>,
          typename Compare = std::less<T>>
class TopKQueue {
private:
  ResizeableArray<T, k> top_k;
  std::priority_queue<T, Container, Compare> remaining_min_heap;
  Compare cmp;
  std::function<bool(const T &, const T &)> r_cmp;

  void insert_top_k(const T &val) {
    auto it = std::lower_bound(top_k.begin(), top_k.end(), val, r_cmp);
    top_k.insert(it, val);
  }

  void push_front(const T &val) { top_k.insert(top_k.begin(), val); }

  void pop_front() { top_k.erase(top_k.begin()); }

public:
  using value_type = T;
  using value_compare = Compare;
  static constexpr int K = k;

  TopKQueue() : r_cmp([this](const T &a, const T &b) { return !cmp(a, b); }) {
    top_k.reserve(k);
  }

  [[nodiscard]] value_compare value_comp() const { return Compare{}; }

  void push(const T &val) {
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

  void pop() {
    if (top_k.empty()) {
      throw std::out_of_range("pop() called on an empty queue");
    }
    pop_front();

    if (!remaining_min_heap.empty()) {
      top_k.push_back(remaining_min_heap.top());
      remaining_min_heap.pop();
    }
  }

  [[nodiscard]] const T &top() const {
    if (top_k.empty()) {
      throw std::out_of_range("top() called on an empty queue");
    }
    return top_k.front();
  }

  T &top() { return const_cast<T &>(std::as_const(*this).top()); }

  T &at(std::size_t i) {
    if (i >= top_k.size()) {
      throw std::out_of_range("at() called with an index out of range");
    }
    return top_k.at(i);
  }

  void remove_at(std::size_t i) {
    if (i >= top_k.size()) {
      throw std::out_of_range("remove_at() called with an index out of range");
    }
    top_k.erase(top_k.begin() + i);

    if (!remaining_min_heap.empty()) {
      top_k.push_back(remaining_min_heap.top());
      remaining_min_heap.pop();
    }
  }

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

template <typename T, int S> inline void print(TopKQueue<T, S> &q) {
  auto &top_k = q.get_top_k();
  for (auto it = top_k.begin(); it != top_k.end(); ++it) {
    std::cout << *it << " ";
  }
}
template <int k> struct TopKQueueHelper {
  template <typename T, typename Container = std::vector<T>,
            typename Compare = std::less<T>>
  using queue_type = TopKQueue<T, k, Container, Compare>;
};

template <QueueConcept Q> std::vector<typename Q::value_type> as_vector(Q &q) {
  typedef typename Q::value_type Value_t;
  // drain store and refill queue
  std::vector<Value_t> elements;
  while (!q.empty()) {
    elements.push_back(q.top_element());
    q.pop();
  }
  for (auto &element : elements) {
    q.push(element);
  }
  return elements;
}

template <WrappedQueueConcept Q>
std::vector<typename Q::element_type> as_vector(Q &q) {
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

template <WrappedQueueConcept Q> void print_table(Q &q) {

  using namespace tabulate;
  using Row_t = Table::Row_t;

  Table table;
  auto elements = as_vector(q);

  // Add headers
  Row_t headers;
  Row_t values;
  Row_t priorities;

  headers.emplace_back("Element");
  values.emplace_back("Value");
  priorities.emplace_back("Priority");

  for (size_t i = 0; i < elements.size(); ++i) {
    headers.emplace_back(std::to_string(i));
    values.emplace_back(std::to_string(elements[i].value));
    priorities.emplace_back(std::to_string(elements[i].priority));
  }
  table.add_row(headers);
  table.add_row(values);
  table.add_row(priorities);

  // Iterator over cells in the first column
  for (auto &cell : table.column(0)) {
    if (cell.get_text() != "Company") {
      cell.format().font_align(FontAlign::right).font_style({FontStyle::bold});
    }
  }

  std::cout << table << std::endl;
}

template <QueueConcept Q> class ContainerIterator {
private:
  std::vector<Q> queues;
  std::vector<bool> active;
  std::size_t active_index;

public:
  ContainerIterator() = default;
  ContainerIterator(std::size_t num_queues) : queues(num_queues) {
    active.resize(num_queues, true);
    active_index = 0;
  }

  Q &get_active_queue() { return queues[active_index]; }
  Q &operator[](std::size_t index) { return queues[index]; }
  Q &at(std::size_t index) { return queues[index]; }

  std::size_t size() { return queues.size(); }
  void set_active_queue(std::size_t index) { active_index = index; }
  std::size_t get_active_index() { return active_index; }

  void deactivate(std::size_t index) { active[index] = false; }

  void next() { active_index = (active_index + 1) % queues.size(); }

  void prev() {
    if (active_index == 0) {
      active_index = queues.size() - 1;
    } else {
      active_index--;
    }
  }
};