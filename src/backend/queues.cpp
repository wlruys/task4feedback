#include "include/queues.hpp"

template <typename T, int s> void ResizeableArray<T, s>::remove_at(int index) {
  for (int i = index; i < l - 1; i++) {
    arr[i] = arr[i + 1];
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::erase(T *it) {
  for (T *i = it; i != arr.begin() + l - 1; i++) {
    *i = *(i + 1);
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::insert(T *it, T val) {
  for (T *i = arr.begin() + l - 1; i >= it; i--) {
    *(i + 1) = *i;
  }
  *it = val;
  l++;
}

template <typename T, int s>
void ResizeableArray<T, s>::insert_at(int index, T val) {
  for (int i = l - 1; i >= index; i--) {
    arr[i + 1] = arr[i];
  }
  arr[index] = val;
  l++;
}

template <typename T, int s> void ResizeableArray<T, s>::pop_front() {
  for (int i = 0; i < l - 1; i++) {
    arr[i] = arr[i + 1];
  }
  l--;
}

template <typename T, int s> void ResizeableArray<T, s>::push_front(T val) {
  for (int i = l - 1; i >= 0; i--) {
    arr[i + 1] = arr[i];
  }
  arr[0] = val;
  l++;
}

template <typename T, template <typename...> class Queue, typename Compare>
consteval int ContainerQueue<T, Queue, Compare>::get_k() {
  if constexpr (is_top_k_queue<QueueType>) {
    return QueueType::K;
  } else {
    return 1;
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
int ContainerQueue<T, Queue, Compare>::topk_size() {
  if constexpr (is_top_k_queue<QueueType>) {
    return pq.topk_size();
  } else {
    return 1;
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
T &ContainerQueue<T, Queue, Compare>::at(std::size_t i) {
  if constexpr (is_top_k_queue<QueueType>) {
    return pq.get_top_k().at(i);
  } else {
    throw std::out_of_range("at() called on a non-top-k queue");
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
void ContainerQueue<T, Queue, Compare>::remove_at(std::size_t i) {
  if constexpr (is_top_k_queue<QueueType>) {
    pq.get_top_k().remove_at(i);
  } else {
    throw std::out_of_range("remove_at() called on a non-top-k queue");
  }
}
template <typename T, template <typename...> class Queue, typename Compare>
auto ContainerQueue<T, Queue, Compare>::get_top_k_elements() {
  if constexpr (is_top_k_queue<QueueType>) {
    return pq.get_top_k();
  } else {
    return std::vector<typename ContainerQueue::element_type>{
        this->top_element()};
  }
}

template <typename T, template <typename...> class Queue, typename Compare>
std::vector<T> ContainerQueue<T, Queue, Compare>::get_top_k() {
  std::vector<T> top_k_values;
  if constexpr (is_top_k_queue<QueueType>) {
    auto &top_k = pq.get_top_k();
    for (auto it = top_k.begin(); it != top_k.end(); ++it) {
      top_k_values.push_back(it->value);
    }
  } else {
    return std::vector<T>{this->top()};
  }
  return top_k_values;
}

template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::insert_top_k(const T &val) {
  auto *it = std::lower_bound(top_k.begin(), top_k.end(), val, r_cmp);
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
  if (top_k.empty()) {
    throw std::out_of_range("pop() called on an empty queue");
  }
  pop_front();

  if (!remaining_min_heap.empty()) {
    top_k.push_back(remaining_min_heap.top());
    remaining_min_heap.pop();
  }
}

template <typename T, int k, typename Container, typename Compare>
const T &TopKQueue<T, k, Container, Compare>::top() const {
  if (top_k.empty()) {
    throw std::out_of_range("top() called on an empty queue");
  }
  return top_k.front();
}

template <typename T, int k, typename Container, typename Compare>
T &TopKQueue<T, k, Container, Compare>::at(std::size_t i) {
  if (i >= top_k.size()) {
    throw std::out_of_range("at() called with an index out of range");
  }
  return top_k.at(i);
}
template <typename T, int k, typename Container, typename Compare>
void TopKQueue<T, k, Container, Compare>::remove_at(std::size_t i) {
  if (i >= top_k.size()) {
    throw std::out_of_range("remove_at() called with an index out of range");
  }
  top_k.erase(top_k.begin() + i);

  if (!remaining_min_heap.empty()) {
    top_k.push_back(remaining_min_heap.top());
    remaining_min_heap.pop();
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

template <typename T> class ActiveIterator {
private:
  std::vector<T> containers;
  std::vector<bool> active;
  std::size_t active_index;
  std::size_t num_active;

public:
  ActiveIterator() = default;
  ActiveIterator(std::size_t num_containers) : containers(num_containers) {
    active.resize(num_containers, true);
    active_index = 0;
    num_active = num_containers;
  }

  bool is_active(std::size_t index) { return active[index]; }
  bool has_active() { return num_active > 0; }

  T &get_active() { return containers[active_index]; }
  T &operator[](std::size_t index) { return containers[index]; }
  T &at(std::size_t index) { return containers[index]; }

  std::size_t size() { return containers.size(); }
  void set_active_queue(std::size_t index) { active_index = index; }
  std::size_t get_active_index() { return active_index; }

  void deactivate(std::size_t index) {
    active[index] = false;
    num_active--;
  }

  void activate(std::size_t index) {
    active[index] = true;
    num_active++;
  }

  void next() { active_index = (active_index + 1) % containers.size(); }

  void prev() {
    if (active_index == 0) {
      active_index = containers.size() - 1;
    } else {
      active_index--;
    }
  }

  void next_active() {
    next();
    while (!active[active_index]) {
      next();
    }
  }

  void prev_active() {
    prev();
    while (!active[active_index]) {
      prev();
    }
  }

  void reset() {
    for (auto &&a : active) {
      a = true;
    }
    num_active = containers.size();
  }
};

template class TopKQueue<int, 3>;
template class TopKQueue<int, 10>;
template class ContainerQueue<int, std::priority_queue>;
template class ActiveIterator<ContainerQueue<int, std::priority_queue>>;