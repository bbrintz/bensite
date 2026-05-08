library(dplyr)
library(glue)
library(jsonlite)
library(purrr)
library(stringr)
library(tidyr)

openalex_author_id <- "A5008421168"
output_path <- "data/collaboration_network_openalex.json"

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

normalize_person_key <- function(x) {
  x |>
    str_to_lower() |>
    str_replace_all("[^a-z]", "")
}

is_ben <- function(name, author_id = NA_character_) {
  id_match <- !is.na(author_id) && author_id %in% c(
    "A5008421168",
    "https://openalex.org/A5008421168",
    "A5121741370",
    "https://openalex.org/A5121741370"
  )

  name_match <- normalize_person_key(name) %in% c(
    "benjbrintz",
    "benbrintz",
    "benjaminbrintz",
    "benjaminjbrintz",
    "bjbrintz",
    "bbrintz"
  )

  id_match || name_match
}

openalex_id <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) return(NA_character_)
  str_remove(x, "^https://openalex.org/")
}

work_source <- function(work) {
  work$primary_location$source$display_name %||% ""
}

included_work <- function(work) {
  type <- work$type %||% ""
  title <- work$display_name %||% ""

  type == "article" && !str_detect(str_to_lower(title), "^author response:")
}

work_url <- function(work) {
  doi <- work$doi %||% NA_character_
  if (!is.na(doi) && nzchar(doi)) return(doi)
  work$id %||% ""
}

message("Fetching OpenAlex works...")
url <- glue(
  "https://api.openalex.org/works?filter=authorships.author.id:{openalex_author_id}",
  "&per-page=200&sort=publication_year:desc",
  "&select=id,doi,display_name,publication_year,cited_by_count,type,authorships,primary_location"
)

response <- fromJSON(url, simplifyVector = FALSE)
works <- response$results

work_records <- imap_dfr(works, function(work, index) {
  tibble(
    work_index = index,
    work_id = openalex_id(work$id),
    title = work$display_name %||% "",
    year = suppressWarnings(as.integer(work$publication_year %||% NA_integer_)),
    cites = suppressWarnings(as.numeric(work$cited_by_count %||% 0)),
    type = work$type %||% "",
    journal = work_source(work),
    url = work_url(work),
    included = included_work(work)
  )
})

included_work_indexes <- work_records |>
  filter(included) |>
  pull(work_index)

author_publications <- imap_dfr(works, function(work, index) {
  authorships <- work$authorships %||% list()

  map_dfr(authorships, function(authorship) {
    author <- authorship$author %||% list()
    author_id <- openalex_id(author$id %||% NA_character_)
    name <- author$display_name %||% authorship$raw_author_name %||% ""

    tibble(
      work_index = index,
      author_id = author_id,
      name = str_squish(name),
      key = if_else(!is.na(author_id) & nzchar(author_id), author_id, normalize_person_key(name))
    )
  })
}) |>
  filter(work_index %in% included_work_indexes) |>
  filter(name != "", key != "") |>
  filter(!map2_lgl(name, author_id, is_ben))

display_names <- author_publications |>
  count(key, name, sort = TRUE) |>
  group_by(key) |>
  arrange(desc(n), desc(str_length(name)), .by_group = TRUE) |>
  slice_head(n = 1) |>
  ungroup() |>
  select(key, name)

top_authors <- author_publications |>
  distinct(key, work_index) |>
  count(key, name = "papers_with_ben", sort = TRUE) |>
  slice_head(n = 20) |>
  left_join(display_names, by = "key")

top_keys <- top_authors$key

work_top_authors <- author_publications |>
  filter(key %in% top_keys) |>
  distinct(work_index, key)

make_pairs <- function(keys) {
  keys <- sort(unique(keys))
  if (length(keys) < 2) return(tibble(source = character(), target = character()))
  pairs <- combn(keys, 2, simplify = FALSE)
  tibble(
    source = map_chr(pairs, 1),
    target = map_chr(pairs, 2)
  )
}

links <- work_top_authors |>
  group_by(work_index) |>
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
  distinct(key, work_index) |>
  left_join(work_records |> filter(included), by = "work_index") |>
  arrange(key, desc(cites), desc(year), title) |>
  group_by(key) |>
  summarize(
    publications = list(
      tibble(
        title = title,
        journal = journal,
        year = year,
        cites = cites,
        url = url
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

collaboration_area <- function(publications) {
  text <- publications |>
    transmute(text = str_to_lower(paste(title, journal))) |>
    pull(text) |>
    paste(collapse = " ")

  case_when(
    str_detect(text, "glaucoma|ophthalmol|cataract|corneal|intraocular|visual field") ~ "Ophthalmology",
    str_detect(text, "contracep|mifepristone|miscarriage") ~ "Reproductive Health",
    str_detect(text, "veteran|caregiver|geriatrics|women & aging|antibiotic resistance|urinary tract") ~ "VA & Health Services",
    str_detect(text, "diarrh|vibrio|bangladesh|mali|global health|travel medicine|rickettsioses|bloody diarrhea|village doctors|serosurveillance") ~ "Global Health & Infectious Disease",
    str_detect(text, "covid|public health|surveillance") ~ "Public Health & Surveillance",
    TRUE ~ "Other Collaborations"
  )
}

nodes <- top_authors |>
  left_join(node_degree, by = "key") |>
  left_join(node_publications, by = "key") |>
  mutate(
    degree = coalesce(degree, 0L),
    weighted_degree = coalesce(weighted_degree, 0L),
    total_cites = coalesce(total_cites, 0),
    radius_score = papers_with_ben + weighted_degree,
    area = map_chr(publications, collaboration_area)
  ) |>
  arrange(desc(papers_with_ben), desc(weighted_degree), name) |>
  transmute(
    id = key,
    name,
    area,
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
  source = "OpenAlex works API",
  source_note = "Counts use OpenAlex records filtered to article-type works, excluding preprints, errata, paratext, peer-review/author-response records, and other duplicate-like publication versions.",
  openalex_author_id = openalex_author_id,
  openalex_record_count = length(works),
  publication_count = length(included_work_indexes),
  author_records = nrow(author_publications),
  nodes = nodes,
  links = links
)

write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, na = "null")
message(glue("Wrote {output_path} with {nrow(nodes)} nodes and {nrow(links)} links."))
