// Free-scrolling log console (pairs with free_log.py).
//
// NiceGUI's stock log component sets the scroll position to the bottom on
// every update: during streaming output that snap repeatedly wins over the
// user's wheel events, so the log cannot be scrolled up while lines keep
// arriving. Here the view follows the tail only while it already rests at
// the bottom; otherwise it is left alone. Scrolling back near the bottom
// resumes following.
const STICK_TOLERANCE_PX = 32;

export default {
  template: `<div ref="qRef" @scroll="onScroll"><slot></slot></div>`,
  data() {
    return {
      pinnedToBottom: true,
      lastScrollHeight: NaN,
    };
  },
  methods: {
    onScroll() {
      const el = this.$el;
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      this.pinnedToBottom = distanceFromBottom < STICK_TOLERANCE_PX;
    },
  },
  beforeUpdate() {
    // The pre-patch scrollHeight is the baseline for detecting how much
    // content this update added or removed; after the patch that is no
    // longer reconstructable.
    this.lastScrollHeight = this.$el ? this.$el.scrollHeight : NaN;
  },
  updated() {
    const el = this.$el;
    if (this.pinnedToBottom) {
      this.$nextTick(() => {
        el.scrollTop = el.scrollHeight;
      });
      return;
    }
    // Appends land below the viewport and must not move it. Trimming at the
    // max_lines cap removes the oldest lines above the viewport, which would
    // drag the text being read upwards by their height; shift back so the
    // visible text stays stationary.
    this.$nextTick(() => {
      if (!Number.isNaN(this.lastScrollHeight)) {
        const heightDelta = el.scrollHeight - this.lastScrollHeight;
        if (heightDelta < 0) el.scrollTop += heightDelta;
      }
    });
  },
};
