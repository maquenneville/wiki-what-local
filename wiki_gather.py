# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:51:57 2023

@author: marca
"""

import re
import mwclient
import mwparserfromhell
import json
import os
from transformers import AutoTokenizer

class WikiGather:
    SECTIONS_TO_IGNORE = [
        "See also",
        "References",
        "External links",
        "Further reading",
        "Footnotes",
        "Bibliography",
        "Sources",
        "Citations",
        "Literature",
        "Footnotes",
        "Notes and references",
        "Photo gallery",
        "Works cited",
        "Photos",
        "Gallery",
        "Notes",
        "References and sources",
        "References and notes",
    ]
    BAD_TITLE_PHRASES = [
        "Wikipedia:",
        "Template:",
        "Template talk:",
        "Help:",
        "Category:",
        "Portal:"
    ]

    def __init__(self, args):
        self.relevant_pages = []
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.wiki_chunks = []
        
        self.page_record = self._load_page_record("wiki_page_record.json")

        self.site = mwclient.Site(
            host='en.wikipedia.org',
            scheme='https',
            path='/w/'
        )

        self.max_tokens = 1080

    def count_tokens(self, text: str) -> int:
        
        tokens = self.tokenizer.tokenize(text)        
        return len(tokens)

    def _save_page_record(self, file_name, input_list):

        data_to_save = [{"title": item.lower()} for item in input_list]
        
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)

    def _load_page_record(self, file_name):

        if not os.path.exists(file_name):
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump([], file, ensure_ascii=False, indent=4)

        try:
            with open(file_name, "r", encoding="utf-8") as file:
                data = json.load(file)
                return [item["title"] for item in data]
        except json.JSONDecodeError:
            return []

    def _is_good_title(self, title):
        for i in self.BAD_TITLE_PHRASES:
            if i in title:
                return False
        return True
    
    def get_wiki_page(self, title: str):

        page = self.site.pages[title]
        
        if not page.exists:
            return None, False
        
        if self._is_disambiguation(page):
            return page, True
        
        return page, False

    def _is_disambiguation(self, page: mwclient.page.Page) -> bool:

        title_lower = page.name.lower()
        if title_lower.endswith("(disambiguation)"):
            return True
        for cat in page.categories():
            if "disambiguation pages" in cat.name.lower():
                return True
        return False

    def find_related_pages(self, title: str, depth=1):

        initial_page, is_disambig = self.get_wiki_page(title)
        if not initial_page:
            print(f"No valid page found for '{title}'.")
            return []
        
        if is_disambig:
            print(f"'{title}' is a disambiguation page.")
            return [initial_page]
        
        linked_pages = self._find_all_pages(initial_page.links(), depth=depth)

        total_pages = [initial_page] + linked_pages
        return total_pages
    
    def _find_all_pages(self, link_pages, depth=1):

        pages = []
        titles_seen = set()

        for link_page in link_pages:
            title_str = link_page.name

            if not self._is_good_title(title_str):
                continue

            title_lower = title_str.lower()

            if title_lower in titles_seen:
                continue
            titles_seen.add(title_lower)
            
            if title_lower in self.page_record:
                continue

            page, is_disambig = self.get_wiki_page(title_str)
            if page:
                print(f"Gathering: {title_str}")
                pages.append(page)
                
                if not is_disambig and depth > 1:
                    sub_links = page.links()  # yields more Page objects
                    sub_pages = self._find_all_pages(sub_links, depth=depth - 1)
                    pages.extend(sub_pages)

        return pages

    def all_subsections_from_section(
        self,
        section: mwparserfromhell.wikicode.Wikicode,
        parent_titles: list[str],
        sections_to_ignore: set[str],
    ) -> list[tuple[list[str], str]]:

        headings = [str(h) for h in section.filter_headings()]
        if not headings:
            return []
        title = headings[0]
        heading_stripped = title.strip("= ").strip()
        if heading_stripped in sections_to_ignore:
            return []
        titles = parent_titles + [title]
        full_text = str(section)
        section_text = full_text.split(title, 1)[1]
        if len(headings) == 1:
            return [(titles, section_text)]
        else:
            first_subtitle = headings[1]
            section_text = section_text.split(first_subtitle, 1)[0]
            results = [(titles, section_text)]
            my_level = len(titles)  # how deeply nested we are
            child_sections = section.get_sections(levels=[my_level + 1])
            for subsection in child_sections:
                results.extend(
                    self.all_subsections_from_section(subsection, titles, sections_to_ignore)
                )
            return results

    def all_subsections_from_title(
        self,
        title: str,
        sections_to_ignore: set[str] = None,
        site_name: str = None,
    ) -> list[tuple[list[str], str]]:

        if sections_to_ignore is None:
            sections_to_ignore = set(self.SECTIONS_TO_IGNORE)

        page = self.site.pages[title]
        text = page.text()
        parsed_text = mwparserfromhell.parse(text)
        headings = [str(h) for h in parsed_text.filter_headings()]
        if headings:
            summary_text = str(parsed_text).split(headings[0], 1)[0]
        else:
            summary_text = str(parsed_text)

        results = [([title], summary_text)]

        top_sections = parsed_text.get_sections(levels=[2])
        for subsection in top_sections:
            results.extend(
                self.all_subsections_from_section(subsection, [title], sections_to_ignore)
            )
        return results

    def clean_section(self, section: tuple[list[str], str]) -> tuple[list[str], str]:

        titles, text = section
        text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)
        text = text.strip()
        return (titles, text)

    def keep_section(self, section: tuple[list[str], str]) -> bool:

        _, text = section
        return len(text) >= 50

    def halved_by_delimiter(self, string: str, delimiter: str = "\n") -> list[str]:

        chunks = string.split(delimiter)
        if len(chunks) == 1:
            return [string, ""]  # no delimiter found at all
        elif len(chunks) == 2:
            return chunks  # already only 2 chunks
        else:
            total_tokens = self.count_tokens(string)
            halfway = total_tokens // 2
            best_diff = halfway
            best_index = -1
            running_string = ""
            for i, chunk in enumerate(chunks):
                left = delimiter.join(chunks[: i + 1])
                left_tokens = self.count_tokens(left)
                diff = abs(halfway - left_tokens)
                if diff > best_diff:
                    # we've overshot
                    break
                else:
                    best_diff = diff
                    best_index = i
                    running_string = left
            if best_index <= 0 or best_index >= len(chunks) - 1:
                # we didn't find a good middle
                return [string, ""]
            left = running_string
            right = delimiter.join(chunks[best_index + 1 :])
            return [left, right]

    # def truncated_string(
    #     self,
    #     string: str,
    #     print_warning: bool = True,
    # ) -> str:

    #     encoded_ids = self.tokenizer.encode(string, add_special_tokens=False)

    #     if len(encoded_ids) > self.max_tokens:
    #         if print_warning:
    #             print(f"Warning: Truncated string from {len(encoded_ids)} to {self.max_tokens} tokens.")
    #         encoded_ids = encoded_ids[:self.max_tokens]

    #     truncated_text = self.tokenizer.decode(encoded_ids, skip_special_tokens=True)
    #     return truncated_text

    # def split_strings_from_subsection(
    #     self,
    #     subsection: tuple[list[str], str],
    #     max_recursion: int = 5,
    # ) -> list[str]:

    #     titles, text = subsection
    #     combined = "\n\n".join(titles + [text])

    #     n_tokens = self.count_tokens(combined)
    #     if n_tokens <= self.max_tokens:
    #         return [combined]
    #     elif max_recursion == 0:
    #         return [self.truncated_string(combined)]
    #     else:

    #         for delimiter in ["\n\n", "\n", ". "]:
    #             splitted = self.halved_by_delimiter(text, delimiter=delimiter)
    #             left, right = splitted
    #             if left == "" or right == "":
    #                 # didn't get a real split, try next delimiter
    #                 continue
    #             # If we get valid left+right, recurse on each side
    #             results = []
    #             for half_text in [left, right]:
    #                 half_subsection = (titles, half_text)
    #                 results.extend(
    #                     self.split_strings_from_subsection(
    #                         half_subsection,
    #                         max_recursion=max_recursion - 1,
    #                     )
    #                 )
    #             return results

    #         # If we couldn't split on any delimiter, fallback to truncation
    #         return [self.truncated_string(combined)]

    def imperfect_split(self, string: str) -> list[str]:

        chunks = []
        remaining_text = string

        while self.count_tokens(remaining_text) > self.max_tokens:
            # Tokenize the text into sentences
            sentences = remaining_text.split(". ")
            current_chunk = []
            current_token_count = 0

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence + ". ")
                if current_token_count + sentence_tokens > self.max_tokens:
                    break
                current_chunk.append(sentence)
                current_token_count += sentence_tokens

            # Save the current chunk and update the remaining text
            chunks.append(". ".join(current_chunk).strip() + ".")
            remaining_text = remaining_text[len(chunks[-1]):].strip()

        # Add the last chunk if any text remains
        if remaining_text:
            chunks.append(remaining_text.strip())

        return chunks


    def split_strings_from_subsection(
        self,
        subsection: tuple[list[str], str],
        max_recursion: int = 5,
    ) -> list[str]:

        titles, text = subsection
        combined = "\n\n".join(titles + [text])

        n_tokens = self.count_tokens(combined)
        if n_tokens <= self.max_tokens:
            return [combined]
        elif max_recursion == 0:
            # Use imperfect split for large text when recursion limit is reached
            chunks = self.imperfect_split(combined)
            return [f"{titles[0]}\n\n{chunk}" for chunk in chunks]

        # Attempt splitting by delimiters
        for delimiter in ["\n\n", "\n", ". "]:
            splitted = self.halved_by_delimiter(text, delimiter=delimiter)
            left, right = splitted
            if left == "" or right == "":
                continue  # Try next delimiter if no valid split

            results = []
            for half_text in [left, right]:
                half_subsection = (titles, half_text)
                results.extend(
                    self.split_strings_from_subsection(
                        half_subsection,
                        max_recursion=max_recursion - 1,
                    )
                )
            return results

        # Fallback to imperfect split if no delimiter split is successful
        chunks = self.imperfect_split(combined)
        return [f"{titles[0]}\n\n{chunk}" for chunk in chunks]



    def gather(self, title: str):

        self.relevant_pages = self.find_related_pages(title)
        for page in self.relevant_pages:

            print(f"Processing sections for: {page.name}")
            try:
                sections = self.all_subsections_from_title(page.name)
            except Exception as e:
                print(f"Error extracting sections for {page.name}: {e}")
                continue

            cleaned = [self.clean_section(s) for s in sections]
            kept = [s for s in cleaned if self.keep_section(s)]

            for s in kept:
                splitted_chunks = self.split_strings_from_subsection(s)
                self.wiki_chunks.extend(splitted_chunks)
        
        self.wiki_chunks = list(set(self.wiki_chunks))

        return "Wiki context gathered"

    def dump(self):

        page_titles = [page.name for page in self.relevant_pages]
        wiki_stuff = self.wiki_chunks.copy()

        self.wiki_chunks = []

        output_path = "doc_store\\wiki_page_record.json"
        self._save_page_record(output_path, page_titles)

        self.page_record = self._load_page_record(output_path)

        self.relevant_pages = []

        return wiki_stuff
