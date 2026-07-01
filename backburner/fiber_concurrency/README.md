# Fiber / concurrency primitives -- backburner

Per project decision (see conversation): dynamic in-place neuron growth via
fiber-based concurrency is deprioritized. Manual layer expansion (allocate a
new, larger buffer and copy) covers current needs; fiber-based growth is
useful for future hardware where RAM itself can grow, not needed now.

The real implementation (fiber.hpp, old_fiber.hpp) lives on the
`cpu_sparse_io` branch, not here -- this file was an orphaned reference
(included fiber.hpp, which doesn't exist in this branch's lib/headers at
all) carried into optim_merge's test folder without its dependency.
When cpu_sparse_io is catalogued, the real fiber.hpp/old_fiber.hpp/
test_fiber.cpp/test_parallel.cpp set should land here together as one
coherent, clearly-labeled unit -- not integrated into the active build.

There was reportedly a working neurogenesis test built on this (see
test_sisldo_neurogenesis.cpp) -- worth preserving for reference even while
backburnered, since it may be directly relevant once dynamic growth is
revisited.
