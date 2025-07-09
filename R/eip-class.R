library(jsonlite)

#' S3 object for the Expectaction-Maximization Algorithm
#'
#' @param X A `(b x c)` matrix representing candidate votes per ballot box.
#'
#' @param W A `(b x g)` matrix representing group votes per ballot box.
#'
#' @param E A `(b x d)` matrix representing the assignment of a ballot box towards a district.
#'
#' @param V A `(d x a)` matrix with the attributes for each district
#'
#' @param json_path A path to a JSON file containing `X`, `W`, `E` and `V` fields, stored as nested arrays. It may contain additional fields with other attributes, which will be added to the returned object.
#'
#' @export
eip <- function(X = NULL, W = NULL, E = NULL, V = NULL, json_path = NULL) {
    # Case when a json is provided
    if (json_path != NULL) {
        if (!all(is.null(X), is.null(W), is.null(E), is.null(V))) {
            stop("If you supply `json_path`, you must NOT supply X, W, E, or V by hand.")
        }
        params <- fromJSON(json_path)
    } else {
        params <- c(X, W, E, V)
    }
    # Validate each value handed
    .validate_eip(params)
    class(params) <- "eip"
    return(params)
}
