#pragma once

// Minimal Tracy stub to allow building without Tracy sources.
#define ZoneScoped
#define ZoneScopedN(x)
#define ZoneScopedNC(x, c)
#define ZoneNamed(x, active)
#define ZoneNamedN(x, name, active)
#define ZoneNamedNC(x, name, c, active)
#define FrameMark
#define FrameMarkNamed(x)
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
#define TracyPlot(name, val)
#define TracyMessage(x, y)
#define TracyMessageL(x)
#define TracyAlloc(ptr, size)
#define TracyFree(ptr)
