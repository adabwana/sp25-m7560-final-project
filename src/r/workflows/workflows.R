# Load necessary libraries
library(workflows)
library(recipes)
library(parsnip)
library(dplyr)

#' Build a Tidymodels Workflow
#'
#' Combines a preprocessing recipe and a parsnip model specification into a
#' tidymodels workflow object.
#'
#' @param recipe A recipe object created by `recipes::recipe()` and potentially
#'   modified with preprocessing steps (e.g., from `create_recipe`).
#' @param model_spec A model specification object created by a `parsnip` function
#'   (e.g., `parsnip::rand_forest()`).
#'
#' @return A workflow object.
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming sample_recipe and mars_spec are defined
#' wf <- build_workflow(recipe = sample_recipe, model_spec = mars_spec)
#' print(wf)
#' }
build_workflow <- function(recipe, model_spec) {
    # Input validation
    if (!inherits(recipe, "recipe")) {
        stop("`recipe` must be a recipe object.", call. = FALSE)
    }
    if (!inherits(model_spec, "model_spec")) {
        stop("`model_spec` must be a model_spec object.", call. = FALSE)
    }

    # Create workflow using the workflows package
    wf <- workflows::workflow() %>%
        workflows::add_recipe(recipe) %>%
        workflows::add_model(model_spec)

    return(wf)
}
