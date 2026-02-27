#pragma once
#include "iterator.hpp"
#include <type_traits>

// Sentinel types for compile-time branch elimination via if constexpr.
// When a drain loop has no breakpoint or no post-success hook, the compiler
// removes those branches entirely — zero runtime overhead.
struct DrainNoOp {
  constexpr void operator()() const noexcept {}
};
struct DrainNoBreakpoint {
  constexpr bool operator()() const noexcept { return false; }
};

// Generic device-queue drain loop with round-robin fairness.
//
// Iterates over per-device queues, cycling devices in round-robin order for
// fairness. A device is "drainable" when it has pending tasks AND has not been
// deactivated due to resource exhaustion this phase. seek_drainable() +
// next_drainable() guarantee active_index always lands on a drainable device,
// so there is no empty-queue check inside the loop body.
//
// Template parameters are resolved at compile time — no std::function, no
// virtual dispatch, all lambdas inlined. Returns true if a breakpoint fired.
//
// Condition  : () -> bool  — continue draining?
// TryAction  : (taskid_t, devid_t) -> bool  — attempt; false deactivates device
// Breakpoint : () -> bool  — stop early? (default: never)
// PostSuccess: () -> void  — hook after a successful pop (default: no-op)
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
