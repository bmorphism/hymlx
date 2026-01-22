;;; HyMLX Telemetry - Toad-Traceable Event System
;;; GF(3) Trit Assignment: ERGODIC (0) - coordination/observation layer
;;;
;;; Provides recordable events for TUI integration with frontier models.
;;; Events are typed by GF(3) trit for balanced tracing.

(import json)
(import time)

(import typing [Any Optional List Dict Callable Union])
(import threading [Lock RLock])
(import collections [deque])

;;; ============================================================
;;; Event Types (GF(3)-Typed)
;;; ============================================================

(defclass TritType []
  "GF(3) trit for event classification."
  (setv PLUS 1)      ; Generation/forward pass
  (setv ERGODIC 0)   ; Coordination/sync
  (setv MINUS -1))   ; Validation/backward pass

(setv (get (globals) "TritType") TritType)

(defclass Event []
  "Immutable telemetry event for toad-traceable recording."
  
  (defn __init__ [self name trit timestamp [payload None] [parent-id None] [event-id None]]
    (setv self.name name)
    (setv self.trit trit)
    (setv self.timestamp timestamp)
    (setv self.payload payload)
    (setv self.parent-id parent-id)
    (setv self.event-id event-id))
  
  (defn __repr__ [self]
    f"Event({self.name!r}, trit={self.trit})"))

;;; ============================================================
;;; Event Ring Buffer (Lock-Free-ish for TUI)
;;; ============================================================

(defclass EventRing []
  "Bounded ring buffer for telemetry events.
  
  Designed for TUI consumption without blocking model inference.
  Uses deque for O(1) append/pop with maxlen enforcement."
  
  (defn __init__ [self [capacity 1024]]
    (setv self._buffer (deque :maxlen capacity))
    (setv self._lock (Lock))
    (setv self._sequence 0))
  
  (defn emit [self event]
    "Append event to ring. Returns sequence number."
    (with [self._lock]
      (setv seq self._sequence)
      (+= self._sequence 1)
      (.append self._buffer #(seq event))
      seq))
  
  (defn drain [self [n None]]
    "Pop up to n events (FIFO). None = all available."
    (setv events [])
    (with [self._lock]
      (setv count (if (is n None) (len self._buffer) (min n (len self._buffer))))
      (for [_ (range count)]
        (when (> (len self._buffer) 0)
          (.append events (.popleft self._buffer)))))
    events)
  
  (defn peek [self [n 10]]
    "View last n events without consuming."
    (with [self._lock]
      (list (get (tuple self._buffer) (slice (- n) None)))))
  
  (defn __len__ [self]
    (len self._buffer)))

;;; ============================================================
;;; Telemetry Context (Thread-Local State)
;;; ============================================================

(defclass TelemetryContext []
  "Global telemetry context with GF(3) conservation tracking."
  
  (defn __init__ [self [ring-capacity 4096]]
    (setv self.ring (EventRing ring-capacity))
    (setv self._trit-sum 0)  ; Running GF(3) balance
    (setv self._lock (RLock))
    (setv self._subscribers [])
    (setv self._enabled True)
    (setv self._event-count 0))
  
  (defn emit [self name trit [payload None] [parent-id None]]
    "Emit a GF(3)-typed event."
    (when (not self._enabled)
      (return None))
    
    (setv ts-hex (format (int (* (time.time) 1000000)) "x"))
    (setv cnt-hex (format self._event-count "04x"))
    (setv event-id f"{ts-hex}-{cnt-hex}")
    (setv event (Event
                  :name name
                  :trit trit
                  :timestamp (time.time)
                  :payload payload
                  :parent-id parent-id
                  :event-id event-id))
    
    (with [self._lock]
      (+= self._event-count 1)
      (setv self._trit-sum (% (+ self._trit-sum trit) 3)))
    
    (setv seq (.emit self.ring event))
    
    ; Notify subscribers (for TUI)
    (for [sub self._subscribers]
      (try
        (sub seq event)
        (except [e Exception]
          None)))  ; Don't let subscriber errors break telemetry
    
    event-id)
  
  (defn subscribe [self callback]
    "Subscribe to events. Callback: (seq, event) -> None"
    (.append self._subscribers callback)
    (fn [] (.remove self._subscribers callback)))  ; Returns unsubscribe fn
  
  (defn gf3-balance [self]
    "Current GF(3) balance. Should converge to 0 for balanced ops."
    self._trit-sum)
  
  (defn enable [self]
    (setv self._enabled True))
  
  (defn disable [self]
    (setv self._enabled False))
  
  (defn export-jsonl [self]
    "Export all events as JSON Lines (for replay)."
    (setv lines [])
    (for [#(seq event) (.drain self.ring)]
      (.append lines (json.dumps {
        "seq" seq
        "name" event.name
        "trit" event.trit
        "timestamp" event.timestamp
        "payload" event.payload
        "parent_id" event.parent-id
        "event_id" event.event-id})))
    (.join "\n" lines)))

;;; ============================================================
;;; Global Context Singleton
;;; ============================================================

(setv _GLOBAL_CTX None)

(defn get-context []
  "Get or create global telemetry context."
  (global _GLOBAL_CTX)
  (when (is _GLOBAL_CTX None)
    (setv _GLOBAL_CTX (TelemetryContext)))
  _GLOBAL_CTX)

(defn reset-context [[capacity 4096]]
  "Reset global context (for testing)."
  (global _GLOBAL_CTX)
  (setv _GLOBAL_CTX (TelemetryContext capacity)))

;;; ============================================================
;;; Event Emission Helpers (GF(3)-Typed)
;;; ============================================================

(defn emit-plus [name [payload None] [parent-id None]]
  "Emit PLUS (+1) event: generation/forward."
  (.emit (get-context) name TritType.PLUS payload parent-id))

(defn emit-ergodic [name [payload None] [parent-id None]]
  "Emit ERGODIC (0) event: coordination/sync."
  (.emit (get-context) name TritType.ERGODIC payload parent-id))

(defn emit-minus [name [payload None] [parent-id None]]
  "Emit MINUS (-1) event: validation/backward."
  (.emit (get-context) name TritType.MINUS payload parent-id))

;;; ============================================================
;;; Tracing Decorators
;;; ============================================================

(defn trace [trit [name None]]
  "Decorator to trace function calls with GF(3) typing."
  (fn [f]
    (setv trace-name (or name f.__name__))
    (fn [#* args #** kwargs]
      (setv start (time.time))
      (setv event-id (.emit (get-context) f"enter:{trace-name}" trit 
                            {"args_count" (len args) "kwargs_keys" (list (.keys kwargs))}))
      (try
        (setv result (f #* args #** kwargs))
        (.emit (get-context) f"exit:{trace-name}" trit
               {"duration_ms" (* 1000 (- (time.time) start))}
               event-id)
        result
        (except [e Exception]
          (.emit (get-context) f"error:{trace-name}" TritType.MINUS
                 {"error" (str e) "type" (. (type e) __name__)}
                 event-id)
          (raise e))))
    ))

(defn trace-plus [[name None]]
  "Trace as PLUS (+1) event."
  (trace TritType.PLUS name))

(defn trace-ergodic [[name None]]
  "Trace as ERGODIC (0) event."
  (trace TritType.ERGODIC name))

(defn trace-minus [[name None]]
  "Trace as MINUS (-1) event."
  (trace TritType.MINUS name))

;;; ============================================================
;;; TUI Protocol (Toad-Traceable Interface)
;;; ============================================================

(defclass ToadProtocol []
  "Protocol for TUI integration with frontier model back-and-forth.
  
  Events flow:
    Model (PLUS) -> Coordination (ERGODIC) -> Validation (MINUS)
    
  TUI subscribes and displays event stream with GF(3) coloring."
  
  (defn __init__ [self ctx]
    (setv self.ctx ctx)
    (setv self._handlers {}))
  
  (defn on [self event-pattern handler]
    "Register handler for event pattern (supports * wildcard)."
    (setv (get self._handlers event-pattern) handler)
    self)
  
  (defn _match [self pattern name]
    "Simple pattern matching with * wildcard."
    (cond
      (= pattern "*") True
      (.endswith pattern "*") (.startswith name (get pattern (slice None -1)))
      (.startswith pattern "*") (.endswith name (get pattern (slice 1 None)))
      True (= pattern name)))
  
  (defn start [self]
    "Start listening to telemetry."
    (defn dispatch [seq event]
      (for [#(pattern handler) (.items self._handlers)]
        (when (self._match pattern event.name)
          (handler seq event))))
    (.subscribe self.ctx dispatch))
  
  (defn format-event [self event [ansi True]]
    "Format event for TUI display with optional ANSI colors."
    (setv trit-colors {1 "\033[32m" 0 "\033[33m" -1 "\033[31m"})  ; green/yellow/red
    (setv reset "\033[0m")
    (setv trit-sym {1 "+" 0 "○" -1 "-"})
    
    (if ansi
      f"{(get trit-colors event.trit)}[{(get trit-sym event.trit)}]{reset} {event.name}"
      f"[{(get trit-sym event.trit)}] {event.name}")))

;;; ============================================================
;;; Model Turn Protocol (for back-and-forth)
;;; ============================================================

(defclass Turn []
  "A model interaction turn with events."
  
  (defn __init__ [self turn-id]
    (setv self.turn-id turn-id)
    (setv self.events [])
    (setv self.start-time (time.time))
    (setv self.end-time None)
    (setv self.parent-turn None))
  
  (defn add-event [self event]
    (.append self.events event))
  
  (defn close [self]
    (setv self.end-time (time.time)))
  
  (defn duration-ms [self]
    (when self.end-time
      (* 1000 (- self.end-time self.start-time)))))

(defclass TurnTracker []
  "Track model turns for conversation-style tracing."
  
  (defn __init__ [self ctx]
    (setv self.ctx ctx)
    (setv self.turns [])
    (setv self.current-turn None)
    (setv self._unsub None))
  
  (defn start-turn [self [parent None]]
    "Begin a new turn."
    (when self.current-turn
      (.close self.current-turn))
    (setv turn-id f"turn-{(len self.turns):04d}")
    (setv self.current-turn (Turn turn-id))
    (setv self.current-turn.parent-turn parent)
    (.append self.turns self.current-turn)
    (emit-ergodic "turn:start" {"turn_id" turn-id "parent" parent})
    turn-id)
  
  (defn end-turn [self]
    "End current turn."
    (when self.current-turn
      (.close self.current-turn)
      (emit-ergodic "turn:end" {
        "turn_id" self.current-turn.turn-id
        "duration_ms" (.duration-ms self.current-turn)
        "event_count" (len self.current-turn.events)}))
    (setv prev self.current-turn)
    (setv self.current-turn None)
    prev)
  
  (defn attach [self]
    "Attach to telemetry context."
    (setv self._unsub (.subscribe self.ctx
      (fn [seq event]
        (when self.current-turn
          (.add-event self.current-turn event))))))
  
  (defn detach [self]
    "Detach from telemetry context."
    (when self._unsub
      (self._unsub)
      (setv self._unsub None)))
  
  (defn export [self]
    "Export turns as structured data."
    (lfor turn self.turns
      {"turn_id" turn.turn-id
       "start" turn.start-time
       "end" turn.end-time
       "duration_ms" (.duration-ms turn)
       "parent" turn.parent-turn
       "event_count" (len turn.events)})))

;;; ============================================================
;;; Replay Support
;;; ============================================================

(defn load-jsonl [jsonl-str]
  "Load events from JSON Lines string."
  (lfor line (.split jsonl-str "\n")
    :if (and line (.strip line))
    (do
      (setv data (json.loads line))
      (Event
        :name (get data "name")
        :trit (get data "trit")
        :timestamp (get data "timestamp")
        :payload (.get data "payload")
        :parent-id (.get data "parent_id")
        :event-id (.get data "event_id")))))

(defn replay-events [events ctx [speed 1.0]]
  "Replay events into context with timing."
  (when (< (len events) 2)
    (return))
  (setv base-time (. (get events 0) timestamp))
  (for [event events]
    (setv delay (/ (- event.timestamp base-time) speed))
    (time.sleep (max 0 delay))
    (.emit ctx event.name event.trit event.payload event.parent-id)
    (setv base-time event.timestamp)))
