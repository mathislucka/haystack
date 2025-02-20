from collections import defaultdict
from typing import Dict, List, Literal, Sequence, Union

import networkx as nx

from haystack import logging
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"azure-ai-formrecognizer>=3.2.0b2\"'") as azure_import:
    from azure.ai.formrecognizer import AnalyzeResult, DocumentLine, DocumentParagraph, DocumentTable, Point

with LazyImport(message="Run 'pip install pandas'") as pandas_import:
    import pandas as pd


if azure_import.is_successful() and pandas_import.is_successful():

    def get_table_content(
        table: "DocumentTable", table_format: Literal["dataframe", "csv"]
    ) -> Union["pd.DataFrame", str]:
        """
        Convert a DocumentTable object to a pandas DataFrame or a CSV string.

        :param table: DocumentTable object from the Azure AI Form Recognizer SDK.
        :param table_format: Format to convert the table to. Either "dataframe" or "csv".
        """
        # Initialize table with empty cells
        table_list = [[""] * table.column_count for _ in range(table.row_count)]

        for cell in table.cells:
            column_span = cell.column_span if cell.column_span else 1
            for c in range(column_span):
                row_span = cell.row_span if cell.row_span else 1
                for r in range(row_span):
                    table_list[cell.row_index + r][cell.column_index + c] = cell.content

        # TODO Re-enable this code once the merge_multiple_column_headers parameter is added to the function
        # # Initialize table with empty cells
        # additional_column_header_rows = set()
        # caption = ""
        # row_idx_start = 0
        #
        # for idx, cell in enumerate(table.cells):
        #     # Remove ':selected:'/':unselected:' tags from cell's content
        #     cell.content = cell.content.replace(":selected:", "")
        #     cell.content = cell.content.replace(":unselected:", "")
        #
        #     # Check if first row is a merged cell spanning whole table
        #     # -> exclude this row and use as a caption
        #     if idx == 0 and cell.column_span == table.column_count:
        #         caption = cell.content
        #         row_idx_start = 1
        #         table_list.pop(0)
        #         continue
        #
        #     column_span = cell.column_span if cell.column_span else 0
        #     for c in range(column_span):  # pylint: disable=invalid-name
        #         row_span = cell.row_span if cell.row_span else 0
        #         for r in range(row_span):  # pylint: disable=invalid-name
        #             if (
        #                 self.merge_multiple_column_headers
        #                 and cell.kind == "columnHeader"
        #                 and cell.row_index > row_idx_start
        #             ):
        #                 # More than one row serves as column header
        #                 table_list[0][cell.column_index + c] += f"\n{cell.content}"
        #                 additional_column_header_rows.add(cell.row_index - row_idx_start)
        #             else:
        #                 table_list[cell.row_index + r - row_idx_start][cell.column_index + c] = cell.content
        #
        # # Remove additional column header rows, as these got attached to the first row
        # for row_idx in sorted(additional_column_header_rows, reverse=True):
        #     del table_list[row_idx]

        if table_format == "csv":
            table_df = pd.DataFrame(data=table_list)
            table_content = table_df.to_csv(header=False, index=False)
        else:
            table_content = pd.DataFrame(columns=table_list[0], data=table_list[1:])

        return table_content

    def check_if_in_table(
        tables_on_page: List["DocumentTable"], line_or_paragraph: Union["DocumentLine", "DocumentParagraph"]
    ) -> bool:
        """
        Check if a DocumentLine or DocumentParagraph is part of a table.

        This is done by comparing the offset of the line or paragraph with the offset of the tables on the page.

        :param tables_on_page: List of DocumentSpan objects representing tables on a given page.
        :param line_or_paragraph: DocumentLine or DocumentParagraph object to check if it is part of a table.
        """
        in_table = False
        for table in tables_on_page:
            table_span = table.spans[0]
            if table_span.offset <= line_or_paragraph.spans[0].offset <= table_span.offset + table_span.length:
                in_table = True
                break
        return in_table

    def remove_objects_contained_within_tables(
        objects_by_page: Dict[int, List[Union["DocumentParagraph", "DocumentLine"]]],
        tables_by_page: Dict[int, List["DocumentTable"]],
    ) -> Dict[int, List[Union["DocumentParagraph", "DocumentLine"]]]:
        """
        Remove all objects that are part of a table from the rows of objects grouped by page.

        :param objects_by_page: Dictionary with page numbers as keys and a list of DocumentLine or DocumentParagraph
            objects as values.
        :param tables_by_page: Dictionary with page numbers as keys and a list of DocumentTable objects as values.
        """
        objects_by_page_without_tables = defaultdict(list)
        for page_number in objects_by_page.keys():  # noqa: PLC0206
            tables_on_page = tables_by_page.get(page_number, [])
            for obj in objects_by_page[page_number]:
                if check_if_in_table(tables_on_page=tables_on_page, line_or_paragraph=obj):
                    continue
                objects_by_page_without_tables[page_number].append(obj)
        return objects_by_page_without_tables

    def insert_tables_into_objects_by_page(
        objects_by_page: Dict[int, List[Union["DocumentParagraph", "DocumentLine"]]],
        tables_by_page: Dict[int, List["DocumentTable"]],
    ) -> Dict[int, List[List[Union["DocumentLine", "DocumentParagraph"]]]]:
        """
        Merge the tables with the objects grouped by page.

        :param objects_by_page: Dictionary with page numbers as keys and a list of DocumentLine or DocumentParagraph
            objects as values.
        :param tables_by_page: Dictionary with page numbers as keys and a list of DocumentTable objects as values.
        """
        objects_by_page_with_tables = defaultdict(list)
        for page_number in objects_by_page.keys():
            objects_on_page = objects_by_page.get(page_number, [])
            tables_on_page = tables_by_page.get(page_number, [])
            objects_by_page_with_tables[page_number] = objects_on_page + tables_on_page
        return objects_by_page_with_tables

    def get_tables_per_page(result: "AnalyzeResult") -> Dict[int, List["DocumentTable"]]:
        """
        Collects the tables from the AnalyzeResult object and groups them by page number.

        :param result: AnalyzeResult object from the Azure Document Intelligence API
        """
        tables_by_page: Dict = defaultdict(list)
        for table in result.tables:
            if table.bounding_regions:
                page_numbers = [b.page_number for b in table.bounding_regions]
            else:
                # If page_number is not available we put the paragraph onto an existing page
                try:
                    current_last_page_number = sorted(tables_by_page.keys())[-1]
                except IndexError:
                    current_last_page_number = 1
                page_numbers = [current_last_page_number]
            tables_by_page[page_numbers[0]].append(table)
        return tables_by_page

    def get_paragraphs_per_page(result: "AnalyzeResult") -> Dict[int, List["DocumentParagraph"]]:
        """
        Groups the paragraphs from the AnalyzeResult object by page number.

        :param result: AnalyzeResult object from the Azure Document Intelligence API.
        """
        paragraphs_by_pages: Dict = defaultdict(list)
        for paragraph in result.paragraphs:
            if paragraph.bounding_regions:
                # If paragraph spans multiple pages we group it with the first page number
                page_numbers = [b.page_number for b in paragraph.bounding_regions]
            else:
                # If page_number is not available we put the paragraph onto an existing page
                try:
                    current_last_page_number = sorted(paragraphs_by_pages.keys())[-1]
                except IndexError:
                    current_last_page_number = 1
                page_numbers = [current_last_page_number]
            paragraphs_by_pages[page_numbers[0]].append(paragraph)
        return paragraphs_by_pages

    def get_lines_per_page(result: "AnalyzeResult") -> Dict[int, List["DocumentLine"]]:
        """
        Groups the lines from the AnalyzeResult object by page number.

        :param result: AnalyzeResult object from the Azure Document Intelligence API.
        """
        lines_by_page: Dict = defaultdict(list)
        for page in result.pages:
            lines = page.lines if page.lines else []
            lines_by_page[page.page_number] = lines
        return lines_by_page

    def get_polygon(obj: Union["DocumentLine", "DocumentParagraph", "DocumentTable"]) -> Sequence["Point"]:
        """
        Get the polygon of a DocumentLine or DocumentParagraph object.

        :param obj: DocumentLine, DocumentParagraph, or DocumentTable object from the Azure AI Form Recognizer SDK.
        """
        if isinstance(obj, DocumentLine):
            return obj.polygon  # type: ignore
        return obj.bounding_regions[0].polygon  # type: ignore

    def get_rows_of_objects_by_page(
        objects_by_page: Dict[int, List[Union["DocumentLine", "DocumentParagraph"]]], threshold_y: float
    ) -> Dict[int, List[List[Union["DocumentLine", "DocumentParagraph"]]]]:
        """
        Groups the lines or paragraphs by rows based on the y-value of the upper left coordinate of their bounding box.

        :param objects_by_page: Dictionary with page numbers as keys and a list of DocumentLine or DocumentParagraph
            objects as values.
        :param threshold_y: Threshold for the y-value difference between the upper left coordinate of the bounding box
            of two lines or paragraphs to be considered part of the same row.
        """
        # Find all pairs of lines that should be grouped together based on the y-value of the upper left coordinate
        # of their bounding box.
        pairs_by_page = defaultdict(list)
        for page_number, objects in objects_by_page.items():
            # Only works if polygons is available
            if all(get_polygon(obj) is not None for obj in objects):
                for i in range(len(objects)):  # pylint: disable=consider-using-enumerate
                    # left_upi, right_upi, right_lowi, left_lowi = get_polygon_fxn(objects[i])
                    left_upi, _, _, _ = get_polygon(objects[i])
                    pairs_by_page[page_number].append([i, i])
                    for j in range(i + 1, len(objects)):  # pylint: disable=invalid-name
                        left_upj, _, _, _ = get_polygon(objects[j])
                        close_on_y_axis = abs(left_upi[1] - left_upj[1]) < threshold_y
                        if close_on_y_axis:
                            pairs_by_page[page_number].append([i, j])
            # Default if polygon is not available
            else:
                logger.warning(
                    "Polygon information for an element on page {pn} is not available so it is not possible to enforce "
                    "a single column page layout.",
                    pn=page_number,
                )
                for i in range(len(objects)):
                    pairs_by_page[page_number].append([i, i])

        # merge the pairs that are connected by page
        merged_pairs_by_page = {}
        for page_number in pairs_by_page:
            graph = nx.Graph()
            graph.add_edges_from(pairs_by_page[page_number])
            merged_pairs_by_page[page_number] = [list(a) for a in list(nx.connected_components(graph))]

        # Convert object indices to the DocumentLine or DocumentParagraph objects
        rows_of_objects_by_page = {}
        for page_number, objects in objects_by_page.items():
            rows = []
            # We use .get(page_idx, []) since the page could be empty
            for row_of_objects in merged_pairs_by_page.get(page_number, []):
                objects_in_row = [objects[object_idx] for object_idx in row_of_objects]
                rows.append(objects_in_row)
            rows_of_objects_by_page[page_number] = rows

        sorted_rows_of_objects_per_page = _get_sorted_rows_of_objects_per_page(
            rows_of_objects_by_page=rows_of_objects_by_page
        )
        return sorted_rows_of_objects_per_page

    def _get_sorted_rows_of_objects_per_page(
        rows_of_objects_by_page: Dict[int, List[List[Union["DocumentLine", "DocumentParagraph"]]]],
    ) -> Dict[int, List]:
        """
        Sort rows of objects per page.

        Sorts the merged objects by page first by the x-value of the upper left bounding box coordinate and then by
        the y-value of the upper left bounding box coordinate.

        :param rows_of_objects_by_page: Dictionary with page numbers as keys and a list of lists of DocumentLine or
            DocumentParagraph objects as values.
        """
        # Sort the merged pairs in each row by the x-value of the upper left bounding box coordinate
        x_sorted_objects_by_page = {}
        for page_number in rows_of_objects_by_page:  # noqa: PLC0206
            sorted_rows = []
            for row_of_objects in rows_of_objects_by_page[page_number]:
                sorted_rows.append(sorted(row_of_objects, key=lambda x: get_polygon(x)[0][0]))  # type: ignore
            x_sorted_objects_by_page[page_number] = sorted_rows

        # Sort each row within the page by the y-value of the upper left bounding box coordinate
        y_sorted_paragraphs_by_page = {}
        for page_number in rows_of_objects_by_page:
            sorted_rows = sorted(
                x_sorted_objects_by_page[page_number],
                key=lambda x: get_polygon(x[0])[0][1],  # type: ignore
            )
            y_sorted_paragraphs_by_page[page_number] = sorted_rows

        return y_sorted_paragraphs_by_page

    def construct_text_from_rows_of_objects_by_page(
        rows_of_objects_by_page: Dict[int, List[List[Union["DocumentLine", "DocumentParagraph", "DocumentTable"]]]],
        result: "AnalyzeResult",
    ) -> str:
        """
        Construct text from the rows of objects grouped by page.

        :param rows_of_objects_by_page: Dictionary with page numbers as keys and a list of lists of DocumentLine or
            DocumentParagraph objects as values.
        :param result: AnalyzeResult object from the Azure Document Intelligence API.
        """
        texts = []
        for page in result.pages:
            page_text = ""
            for row_of_objects in rows_of_objects_by_page.get(page.page_number, []):
                for obj in row_of_objects:
                    if isinstance(obj, DocumentTable):
                        table_content = get_table_content(table=obj, table_format="csv")
                        page_text += table_content + " "
                    else:
                        page_text += obj.content + " "
                    page_text = page_text[:-1]
                page_text += "\n"
            texts.append(page_text)
        all_text = "\f".join(texts)
        return all_text
