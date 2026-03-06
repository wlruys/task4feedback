#pragma once
#include "iterator.hpp"
#include <type_traits>

struct DrainNoOp {
  constexpr void operator()() const noexcept {}
};
struct DrainNoBreakpoint {
  constexpr bool operator()() const noexcept { return false; }
};

template <typename Q, typename Condition, typename TryAction,
          typename Breakpoint = DrainNoBreakpoint, typename PostSuccess = DrainNoOp>
[[nodiscard]] inline bool drain_device_queue(Q &queue, Condition &&condition,
                                             TryAction &&try_action,
                                             Breakpoint &&check_breakpoint = {},
                                             PostSuccess &&post_success = {}) {
  queue.reset();
  queue.seek_drainable();

  while (queue.has_active() && condition()) {
    if constexpr (!std::is_same_v<std::decay_t<Breakpoint>, DrainNoBreakpoint>) {
      if (check_breakpoint()) return true;
    }

    taskid_t task_id = queue.top();
    auto device_id = static_cast<devid_t>(queue.get_active_index());

    if (!try_action(task_id, device_id)) {
      queue.deactivate();
      queue.next_drainable();
      continue;
    }

    queue.pop(); // clears tasks_mask bit if the queue becomes empty
    if constexpr (!std::is_same_v<std::decay_t<PostSuccess>, DrainNoOp>) {
      post_success();
    }
    queue.next_drainable();
  }

  return false;
}
