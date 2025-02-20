# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.azure_ocr_utils import (
    construct_text_from_rows_of_objects_by_page,
    get_lines_per_page,
    get_paragraphs_per_page,
    get_rows_of_objects_by_page,
    get_table_content,
    get_tables_per_page,
    insert_tables_into_objects_by_page,
    remove_objects_contained_within_tables,
)

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"azure-ai-formrecognizer>=3.2.0b2\"'") as azure_import:
    from azure.ai.formrecognizer import AnalysisFeature, AnalyzeResult, DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential


@component
class AzureOCRDocumentConverter:
    """
    Converts files to documents using Azure's Document Intelligence service.

    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    To use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

    ### Usage example

    ```python
    from haystack.components.converters import AzureOCRDocumentConverter
    from haystack.utils import Secret

    converter = AzureOCRDocumentConverter(endpoint="<url>", api_key=Secret.from_token("<your-api-key>"))
    results = converter.run(sources=["path/to/doc_with_images.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        endpoint: str,
        api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"),
        model_id: str = "prebuilt-read",
        preceding_context_len: int = 3,
        following_context_len: int = 3,
        merge_multiple_column_headers: bool = True,
        page_layout: Literal[
            "natural", "single_column_by_line", "single_column_by_paragraph", "single_column"
        ] = "natural",
        threshold_y: Optional[float] = 0.05,
        store_full_path: bool = False,
        *,
        analysis_features: Optional[List[str]] = None,
        table_format: Literal["csv", "text"] = "csv",
        extract_tables_separately: bool = True,
    ):
        """
        Creates an AzureOCRDocumentConverter component.

        :param endpoint:
            The endpoint of your Azure resource.
        :param api_key:
            The API key of your Azure resource.
        :param model_id:
            The ID of the model you want to use. For a list of available models, see [Azure documentation]
            (https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature).
        :param preceding_context_len: Number of lines before a table to include as preceding context
            (this will be added to the metadata).
        :param following_context_len: Number of lines after a table to include as subsequent context (
            this will be added to the metadata).
        :param merge_multiple_column_headers: If `True`, merges multiple column header rows into a single row.
        :param page_layout: The type reading order to follow. Possible options:
            - `natural`: Uses the natural reading order determined by Azure.
            - `single_column`: Groups all lines with the same height on the page based on a threshold
            determined by `threshold_y`.
        :param page_layout: The type reading order to follow.
            - "natural" means the natural reading order determined by Azure will be used
            - "single_column_by_line" means all lines with the same height on the page will be grouped together
            based on a threshold determined by `threshold_y`
            - "single_column_by_paragraph" means all paragraphs with the same height on the page will be grouped
            together based on a threshold determined by `threshold_y`
            - "single_column" is the same as "single_column_by_line" and is kept for backwards compatibility
        :param threshold_y: The threshold to determine if two recognized elements in a PDF should be grouped into a
            single line. This is especially relevant for section headers or numbers which may be spatially separated
            on the horizontal axis from the remaining text. The threshold is specified in units of inches.
            This is only relevant if "single_column" is chosen for `page_layout`.
        :param analysis_features: Additional document analysis features to enable for Azure Form Recognizer.
            They include "ocrHighResolution", "languages", "barcodes", "formulas", "keyValuePairs", "styleFont".
        :param table_format: The format in which the tables should be returned. Options are "csv" or "text".
        :param extract_tables_separately: Whether to save detected tables from the document separately in a tabular
            format determined by `table_format`. If set to False, tables will be included with the other text.
        """
        azure_import.check()

        self.analysis_features = analysis_features
        _analysis_features = None
        if analysis_features:
            _analysis_features = [AnalysisFeature(feature) for feature in analysis_features]
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key.resolve_value() or ""), features=_analysis_features
        )  # type: ignore
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key
        self.preceding_context_len = preceding_context_len
        self.following_context_len = following_context_len
        self.merge_multiple_column_headers = merge_multiple_column_headers
        self.page_layout = page_layout
        self.threshold_y = threshold_y
        self.store_full_path = store_full_path
        if self.page_layout == "single_column" and self.threshold_y is None:
            self.threshold_y = 0.05

        self.table_format = table_format
        self.extract_tables_separately = extract_tables_separately
        if self.extract_tables_separately and self.table_format == "text":
            raise ValueError("Table format 'text' is not supported when extracting tables separately. Choose 'csv'.")

    @component.output_types(documents=List[Document], raw_azure_response=List[Dict])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Convert a list of files to Documents using Azure's Document Intelligence service.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be
            zipped. If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents
            - `raw_azure_response`: List of raw Azure responses used to create the Documents
        """
        documents = []
        azure_output = []
        meta_list: List[Dict[str, Any]] = normalize_metadata(meta=meta, sources_count=len(sources))
        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            poller = self.document_analysis_client.begin_analyze_document(
                model_id=self.model_id, document=bytestream.data
            )
            result = poller.result()
            azure_output.append(result.to_dict())

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)
            docs = self._convert_tables_and_text(result=result, meta=merged_metadata)
            documents.extend(docs)

        return {"documents": documents, "raw_azure_response": azure_output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            endpoint=self.endpoint,
            model_id=self.model_id,
            preceding_context_len=self.preceding_context_len,
            following_context_len=self.following_context_len,
            merge_multiple_column_headers=self.merge_multiple_column_headers,
            page_layout=self.page_layout,
            threshold_y=self.threshold_y,
            store_full_path=self.store_full_path,
            analysis_features=self.analysis_features,
            table_format=self.table_format,
            extract_tables_separately=self.extract_tables_separately,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOCRDocumentConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    # pylint: disable=line-too-long
    def _convert_tables_and_text(self, result: "AnalyzeResult", meta: Optional[Dict[str, Any]]) -> List[Document]:
        """
        Converts the tables and text extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
            Can be any custom keys and values.
        :returns: List of Documents containing the tables and text extracted from the AnalyzeResult object.
        """
        if self.extract_tables_separately:
            tables = self._extract_table_documents(result=result, meta=meta)
        else:
            tables = []

        if self.page_layout == "natural":
            text = self._extract_text_doc_with_natural_ordering(result=result, meta=meta)
        elif self.page_layout in {"single_column_by_line", "single_column"}:
            text = self._extract_text_doc_with_single_column_ordering(
                result=result, object_type="line", meta=meta, threshold_y=self.threshold_y
            )
        else:
            text = self._extract_text_doc_with_single_column_ordering(
                result=result, object_type="paragraph", meta=meta, threshold_y=self.threshold_y
            )
        return [*tables, text]

    def _extract_table_documents(self, result: "AnalyzeResult", meta: Optional[Dict[str, Any]]) -> List[Document]:
        """
        Converts the tables extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult Azure object
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.

        :returns: List of Documents containing the tables extracted from the AnalyzeResult object.
        """
        converted_tables: List[Document] = []

        if not result.tables:
            return converted_tables

        for table in result.tables:
            table_content = get_table_content(table=table, table_format=self.table_format)  # type: ignore

            # Get preceding context of table
            if table.bounding_regions:
                table_beginning_page = next(
                    page for page in result.pages if page.page_number == table.bounding_regions[0].page_number
                )
            else:
                table_beginning_page = None
            table_start_offset = table.spans[0].offset
            if table_beginning_page and table_beginning_page.lines:
                preceding_lines = [
                    line.content for line in table_beginning_page.lines if line.spans[0].offset < table_start_offset
                ]
            else:
                preceding_lines = []
            preceding_context = "\n".join(preceding_lines[-self.preceding_context_len :])
            preceding_context = preceding_context.strip()

            # Get following context
            if table.bounding_regions and len(table.bounding_regions) == 1:
                table_end_page = table_beginning_page
            elif table.bounding_regions:
                table_end_page = next(
                    page for page in result.pages if page.page_number == table.bounding_regions[-1].page_number
                )
            else:
                table_end_page = None

            table_end_offset = table_start_offset + table.spans[0].length
            if table_end_page and table_end_page.lines:
                following_lines = [
                    line.content for line in table_end_page.lines if line.spans[0].offset > table_end_offset
                ]
            else:
                following_lines = []
            following_context = "\n".join(following_lines[: self.following_context_len])

            if meta is None:
                meta = {}
            table_meta = {**meta, "preceding_context": preceding_context, "following_context": following_context}

            if table.bounding_regions:
                table_meta["page"] = table.bounding_regions[0].page_number

            converted_tables.append(Document(content=table_content, meta=table_meta))

        # Sort by page number
        converted_tables = sorted(converted_tables, key=lambda x: x.meta.get("page", 0))
        return converted_tables

    def _extract_text_doc_with_single_column_ordering(
        self,
        result: "AnalyzeResult",
        object_type: Literal["line", "paragraph"],
        meta: Optional[Dict[str, str]],
        threshold_y: float = 0.05,
    ) -> Document:
        """
        Converts the text extracted by Azure's Document Intelligence service.

        This converts the `AnalyzeResult` object into a single Haystack Document. We add "\f" separators between pages
        to differentiate between the text on separate pages.

        :param result: The AnalyzeResult object returned by the `DocumentAnalysisClient.begin_analyze_document` method.
        :param object_type: Type of objects to group. Either "line" or "paragraph".
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
        :param threshold_y: Threshold for the y-value difference between the upper left coordinate of the bounding box
            of two paragraphs to be considered part of the same row.
        """
        tables_by_page = get_tables_per_page(result=result)

        if object_type == "line":
            objects_by_page = get_lines_per_page(result=result)
        else:
            objects_by_page = get_paragraphs_per_page(result=result)

        if self.extract_tables_separately:
            objects_by_page = remove_objects_contained_within_tables(
                objects_by_page=objects_by_page, tables_by_page=tables_by_page
            )
        elif self.table_format == "csv":
            objects_by_page = remove_objects_contained_within_tables(
                objects_by_page=objects_by_page, tables_by_page=tables_by_page
            )
            objects_by_page = insert_tables_into_objects_by_page(
                objects_by_page=objects_by_page, tables_by_page=tables_by_page
            )

        rows_of_objects_by_page = get_rows_of_objects_by_page(objects_by_page=objects_by_page, threshold_y=threshold_y)

        all_text = construct_text_from_rows_of_objects_by_page(
            rows_of_objects_by_page=rows_of_objects_by_page, result=result
        )
        return Document(content=all_text, meta=meta)

    def _extract_text_doc_with_natural_ordering(
        self, result: "AnalyzeResult", meta: Optional[Dict[str, str]]
    ) -> Document:
        """
        Converts the text extracted by Azure's Document Intelligence service.

        This converts the `AnalyzeResult` object into a single Haystack Document. We add "\f" separators between to
        differentiate between the text on separate pages.

        :param result: The AnalyzeResult object returned by the `DocumentAnalysisClient.begin_analyze_document` method.
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
        """
        tables_by_page = get_tables_per_page(result=result)
        paragraphs_by_page = get_paragraphs_per_page(result=result)

        if self.extract_tables_separately:
            paragraphs_by_page = remove_objects_contained_within_tables(
                objects_by_page=paragraphs_by_page, tables_by_page=tables_by_page
            )
        elif self.table_format == "csv":
            paragraphs_by_page = remove_objects_contained_within_tables(
                objects_by_page=paragraphs_by_page, tables_by_page=tables_by_page
            )
            paragraphs_by_page = insert_tables_into_objects_by_page(
                objects_by_page=paragraphs_by_page, tables_by_page=tables_by_page
            )

        rows_of_objects_by_page: Dict = defaultdict(list)
        for page in result.pages:
            paragraphs_on_page = paragraphs_by_page.get(page.page_number, [])
            sorted_objects_on_page = sorted(paragraphs_on_page, key=lambda x: x.spans[0].offset)
            for obj in sorted_objects_on_page:
                rows_of_objects_by_page[page.page_number].append([obj])

        all_text = construct_text_from_rows_of_objects_by_page(
            rows_of_objects_by_page=rows_of_objects_by_page, result=result
        )
        return Document(content=all_text, meta=meta)
