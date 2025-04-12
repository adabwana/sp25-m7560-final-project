(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(def r-host "r")  ; container name becomes hostname
(def python-host "python")


(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "M7550: Final Project"}
    :subdirs-to-sync     ["notebooks" "data" 
                          "presentation" "images" "presentation/images"]
    :source-path         ["src/index.clj"
                          "notebooks/report/feature_engineering.md"
                          "notebooks/report/models_overview.md"
                          ;; "notebooks/report/model_training.md"
                          ;; "notebooks/report/model_testing.md"
                          ;; "notebooks/report/eval_framework.md"
                          "notebooks/report/eval.md"
                          "notebooks/report/predictions.md"
                          "notebooks/report/appendix.md"]
    :base-target-path    "docs"
    :clean-up-target-dir true
    :engines             {:r      {:host r-host
                                   :port 6311}
                          :python  {:host python-host
                                    :port 8888}}}))

(comment
  (build))
