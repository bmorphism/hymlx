;;; HyMLX CLI - Hy-native command interface
;;; GF(3) Trit: ERGODIC (0) - coordination layer

(import argparse)
(import time)
(import sys)

;;; ============================================================
;;; Telemetry Commands
;;; ============================================================

(defn cmd-telemetry-watch [args]
  "Watch telemetry events in real-time (TUI mode)."
  (import hymlx.telemetry :as tele)
  
  (print "📡 HyMLX Telemetry Watch")
  (print "Listening for events... (Ctrl+C to stop)\n")
  
  (setv ctx (tele.get-context))
  (setv toad (tele.ToadProtocol ctx))
  
  (defn on-event [seq event]
    (print (+ "[" (str seq) "] " (.format-event toad event))))
  
  (.on toad "*" on-event)
  (.start toad)
  
  (try
    (while True
      (time.sleep 0.1))
    (except [KeyboardInterrupt]
      (print "\n✅ Watch stopped."))))

(defn cmd-telemetry-demo [args]
  "Run demo with traced transforms."
  (import mlx.core :as mx)
  (import hymlx.transforms :as tf)
  (import hymlx.telemetry :as tele)
  
  (print "🔬 Telemetry Demo\n")
  
  (tele.reset-context)
  (setv ctx (tele.get-context))
  
  ;; Emit custom GF(3)-typed events
  (print "1. Custom GF(3)-typed events:")
  (tele.emit-plus "inference:start" {"model" "test"})
  (tele.emit-ergodic "sync:checkpoint")
  (tele.emit-minus "validation:loss" {"value" 0.5})
  
  ;; Show balance
  (print (+ "\n📊 GF(3) Balance: " (str (.gf3-balance ctx))))
  
  ;; Show events
  (print "\n2. Captured events:")
  (setv events (.peek ctx.ring 5))
  (for [#(seq event) events]
    (print (+ "  [" (str seq) "] " event.name " (trit=" (str event.trit) ")")))
  
  (print "\n✅ Demo complete!"))

(defn cmd-telemetry-export [args]
  "Export events as JSON Lines."
  (import hymlx.telemetry :as tele)
  
  (setv ctx (tele.get-context))
  (setv jsonl (.export-jsonl ctx))
  
  (if args.output
    (do
      (with [f (open args.output "w")]
        (.write f jsonl))
      (print (+ "Exported to " args.output)))
    (print jsonl)))

;;; ============================================================
;;; Turn Tracking Commands
;;; ============================================================

(defn cmd-turns-start [args]
  "Start a new model turn."
  (import hymlx.telemetry :as tele)
  
  (setv ctx (tele.get-context))
  (setv tracker (tele.TurnTracker ctx))
  (.attach tracker)
  
  (setv turn-id (.start-turn tracker args.parent))
  (print (+ "Started turn: " turn-id)))

(defn cmd-turns-end [args]
  "End current turn."
  (import hymlx.telemetry :as tele)
  
  (setv ctx (tele.get-context))
  (setv tracker (tele.TurnTracker ctx))
  (.attach tracker)
  
  (setv turn (.end-turn tracker))
  (if turn
    (print (+ "Ended turn: " turn.turn-id))
    (print "No active turn")))

;;; ============================================================
;;; Main Entry Point  
;;; ============================================================

(defn main []
  "HyMLX Hy-native CLI."
  (setv parser (argparse.ArgumentParser
    :description "HyMLX: Hy-native CLI for MLX transforms"))
  
  (setv subparsers (.add-subparsers parser :dest "command"))
  
  ;; telemetry command group
  (setv tele-parser (.add-parser subparsers "telemetry" :help "Telemetry commands"))
  (setv tele-subs (.add-subparsers tele-parser :dest "tele_cmd"))
  
  (.add-parser tele-subs "watch" :help "Watch events (TUI)")
  (.add-parser tele-subs "demo" :help "Run telemetry demo")
  
  (setv export-parser (.add-parser tele-subs "export" :help "Export JSONL"))
  (.add-argument export-parser "-o" "--output" :help "Output file")
  
  ;; turns command group
  (setv turns-parser (.add-parser subparsers "turns" :help "Turn tracking"))
  (setv turns-subs (.add-subparsers turns-parser :dest "turns_cmd"))
  
  (setv start-parser (.add-parser turns-subs "start" :help "Start turn"))
  (.add-argument start-parser "-p" "--parent" :help "Parent turn ID")
  
  (.add-parser turns-subs "end" :help "End turn")
  
  (setv args (.parse-args parser))
  
  (cond
    (= args.command "telemetry")
      (cond
        (= args.tele_cmd "watch") (cmd-telemetry-watch args)
        (= args.tele_cmd "demo") (cmd-telemetry-demo args)
        (= args.tele_cmd "export") (cmd-telemetry-export args)
        True (.print-help tele-parser))
    (= args.command "turns")
      (cond
        (= args.turns_cmd "start") (cmd-turns-start args)
        (= args.turns_cmd "end") (cmd-turns-end args)
        True (.print-help turns-parser))
    True (.print-help parser)))

(when (= __name__ "__main__")
  (main))
