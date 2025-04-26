(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "M7560: Final Project"}
    :subdirs-to-sync     ["notebooks" "data"
                          "presentation" "images" "presentation/images"]
    :source-path         ["src/index.clj"
                          "notebooks/report/feature_engineering.md"
                          "notebooks/report/features_external.md"
                          "notebooks/report/preprocessing.md"
                          "notebooks/report/model_tuning.md"
                          "notebooks/report/neural_networks.md"
                          "notebooks/report/eval.md"
                          "notebooks/report/predictions.md"
                          "notebooks/report/kmeans.md"]
    :base-target-path    "docs"
    :clean-up-target-dir true}))

(comment
  (build))
