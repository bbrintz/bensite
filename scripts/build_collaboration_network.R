library(dplyr)
library(glue)
library(jsonlite)
library(purrr)
library(scholar)
library(stringr)
library(tidyr)

scholar_id <- "nYA8PAsAAAAJ"
output_path <- "data/scholar_collaboration_network.json"

normalize_person_key <- function(x) {
  x |>
    str_to_lower() |>
    str_replace_all("[^a-z]", "")
}

split_people <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(x)) return(character())
  x |>
    str_replace_all("\\s+", " ") |>
    str_replace("\\s+\\.\\.\\.$", "") |>
    str_split("\\s*,\\s*") |>
    unlist(use.names = FALSE) |>
    str_squish() |>
    discard(~ .x == "" || .x == "...")
}

is_ben <- function(x) {
  normalize_person_key(x) %in% c(
    "benjbrintz",
    "benbrintz",
    "benjaminbrintz",
    "benjaminjbrintz",
    "bjbrintz",
    "bbrintz"
  )
}

message("Fetching Google Scholar publication list...")
pubs <- get_publications(scholar_id, pagesize = 100)

authors_for_pub <- function(pubid, fallback, index, total) {
  message(glue("[{index}/{total}] authors for {pubid}"))

  full <- tryCatch(
    get_complete_authors(scholar_id, pubid, delay = 0.12),
    error = function(e) NA_character_
  )

  full_names <- if (!is.na(full) && !is.null(names(full)) && nzchar(names(full))) {
    split_people(names(full))
  } else {
    character()
  }

  if (length(full_names) > 0) {
    return(full_names)
  }

  split_people(fallback)
}

pub_records <- pubs |>
  mutate(
    pub_index = row_number(),
    year = suppressWarnings(as.integer(year)),
    cites = suppressWarnings(as.numeric(cites)),
    authors = pmap(
      list(pubid, author, pub_index),
      ~ authors_for_pub(..1, ..2, ..3, nrow(pubs))
    )
  )

author_publications <- pub_records |>
  select(pub_index, title, journal, year, cites, pubid, authors) |>
  unnest_longer(authors, values_to = "name") |>
  mutate(
    name = str_squish(name),
    key = normalize_person_key(name)
  ) |>
  filter(name != "", !is_ben(name), key != "")

display_names <- author_publications |>
  count(key, name, sort = TRUE) |>
  group_by(key) |>
  arrange(desc(n), desc(str_length(name)), .by_group = TRUE) |>
  slice_head(n = 1) |>
  ungroup() |>
  select(key, name)

top_authors <- author_publications |>
  distinct(key, pub_index) |>
  count(key, name = "papers_with_ben", sort = TRUE) |>
  slice_head(n = 20) |>
  left_join(display_names, by = "key")

top_keys <- top_authors$key

pub_top_authors <- author_publications |>
  filter(key %in% top_keys) |>
  distinct(pub_index, key)

make_pairs <- function(keys) {
  keys <- sort(unique(keys))
  if (length(keys) < 2) return(tibble(source = character(), target = character()))
  pairs <- combn(keys, 2, simplify = FALSE)
  tibble(
    source = map_chr(pairs, 1),
    target = map_chr(pairs, 2)
  )
}

links <- pub_top_authors |>
  group_by(pub_index) |>
  summarize(pairs = list(make_pairs(key)), .groups = "drop") |>
  unnest(pairs) |>
  count(source, target, name = "weight") |>
  filter(weight > 0)

node_degree <- bind_rows(
  links |> transmute(key = source, neighbor = target, weight),
  links |> transmute(key = target, neighbor = source, weight)
) |>
  group_by(key) |>
  summarize(
    degree = n_distinct(neighbor),
    weighted_degree = sum(weight),
    .groups = "drop"
  )

node_publications <- author_publications |>
  filter(key %in% top_keys) |>
  distinct(key, title, journal, year, cites, pubid) |>
  arrange(key, desc(cites), desc(year), title) |>
  group_by(key) |>
  summarize(
    publications = list(
      tibble(
        title = title,
        journal = journal,
        year = year,
        cites = cites,
        url = glue("https://scholar.google.com/citations?view_op=view_citation&user={scholar_id}&citation_for_view={scholar_id}:{pubid}")
      ) |>
        slice_head(n = 8)
    ),
    total_cites = sum(cites, na.rm = TRUE),
    first_year = suppressWarnings(min(year, na.rm = TRUE)),
    last_year = suppressWarnings(max(year, na.rm = TRUE)),
    .groups = "drop"
  ) |>
  mutate(
    first_year = ifelse(is.infinite(first_year), NA_integer_, first_year),
    last_year = ifelse(is.infinite(last_year), NA_integer_, last_year)
  )

nodes <- top_authors |>
  left_join(node_degree, by = "key") |>
  left_join(node_publications, by = "key") |>
  mutate(
    degree = coalesce(degree, 0L),
    weighted_degree = coalesce(weighted_degree, 0L),
    total_cites = coalesce(total_cites, 0),
    radius_score = papers_with_ben + weighted_degree
  ) |>
  arrange(desc(papers_with_ben), desc(weighted_degree), name) |>
  transmute(
    id = key,
    name,
    papers = papers_with_ben,
    degree,
    weighted_degree,
    total_cites,
    first_year,
    last_year,
    radius_score,
    publications
  )

payload <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  source = "Google Scholar via R package scholar",
  scholar_id = scholar_id,
  publication_count = nrow(pubs),
  author_records = nrow(author_publications),
  nodes = nodes,
  links = links
)

write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, na = "null")
message(glue("Wrote {output_path} with {nrow(nodes)} nodes and {nrow(links)} links."))
