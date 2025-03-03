# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
import pytest

from openai.types.chat import ChatCompletionChunk
from openai.types.chat import chat_completion_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret
from haystack.components.generators.chat.openai import OpenAIChatGenerator


class TestOpenAIChatGeneratorStreamOptions:
    """
    Tests for handling OpenAI stream_options, particularly the include_usage option
    that causes empty choices arrays in some stream chunks.
    """

    def test_stream_response_with_empty_choices(self):
        """
        Test that _handle_stream_response correctly handles chunks with empty choices arrays.
        This simulates what happens when stream_options.include_usage is set to True.
        """
        # Create a component instance
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        
        # Create a mock Stream object that will yield chunks with and without choices
        mock_stream = MagicMock()
        
        # Create two chunks - one with empty choices and one with a normal choice
        usage_chunk = ChatCompletionChunk(
            id="usage-chunk",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            choices=[],  # Empty choices array for usage stats
            created=1234567890,
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        
        normal_chunk = ChatCompletionChunk(
            id="normal-chunk",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        content="Hello world", 
                        role="assistant",
                    ),
                )
            ],
            created=1234567890,
        )
        
        # Make the stream yield both chunks in sequence
        mock_stream.__iter__ = lambda self: iter([usage_chunk, normal_chunk])
        
        # Mock callback function
        callback = MagicMock()
        
        # Call the method
        result = component._handle_stream_response(mock_stream, callback)
        
        # Check that the callback was called exactly once (only for the normal chunk)
        assert callback.call_count == 1
        
        # Verify the result contains the expected message
        assert len(result) == 1
        assert isinstance(result[0], ChatMessage)
        assert result[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_async_stream_response_with_empty_choices(self):
        """
        Test that _handle_async_stream_response correctly handles chunks with empty choices arrays.
        This simulates what happens when stream_options.include_usage is set to True.
        """
        # Create a component instance
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        
        # Create an async generator that will yield chunks with and without choices
        async def mock_async_stream():
            # Yield a chunk with empty choices (usage stats)
            yield ChatCompletionChunk(
                id="usage-chunk",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                choices=[],  # Empty choices array for usage stats
                created=1234567890,
                usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
            )
            
            # Yield a normal chunk with content
            yield ChatCompletionChunk(
                id="normal-chunk",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                choices=[
                    chat_completion_chunk.Choice(
                        finish_reason="stop",
                        index=0,
                        delta=chat_completion_chunk.ChoiceDelta(
                            content="Hello world", 
                            role="assistant",
                        ),
                    )
                ],
                created=1234567890,
            )
        
        # Create mock AsyncStream
        mock_stream = MagicMock()
        mock_stream.__aiter__ = mock_async_stream
        
        # Mock async callback function
        async def async_callback(chunk):
            pass
            
        callback = MagicMock(side_effect=async_callback)
        
        # Call the method
        result = await component._handle_async_stream_response(mock_stream, callback)
        
        # Check that the callback was called exactly once (only for the normal chunk)
        assert callback.call_count == 1
        
        # Verify the result contains the expected message
        assert len(result) == 1
        assert isinstance(result[0], ChatMessage)
        assert result[0].text == "Hello world"

    @patch("openai.resources.chat.completions.Completions.create")
    def test_streaming_with_include_usage(self, mock_create):
        """
        Test a full run with generation_kwargs including stream_options.include_usage
        """
        # Create a component instance
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={
                "stream_options": {
                    "include_usage": True
                }
            }
        )
        
        # Setup the mock to return a stream object that yields chunks
        mock_stream = MagicMock()
        
        # Create chunks including one with empty choices array
        usage_chunk = ChatCompletionChunk(
            id="usage-chunk",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            choices=[],  # Empty choices array for usage stats
            created=1234567890,
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        
        content_chunk = ChatCompletionChunk(
            id="content-chunk",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        content="This is a test response", 
                        role="assistant",
                    ),
                )
            ],
            created=1234567890,
        )
        
        # Make the stream yield both chunks in sequence
        mock_stream.__iter__ = lambda self: iter([usage_chunk, content_chunk])
        mock_create.return_value = mock_stream
        
        # Setup streaming callback
        callback_chunks = []
        def streaming_callback(chunk: StreamingChunk):
            callback_chunks.append(chunk)
        
        # Run with streaming callback
        result = component.run(
            messages=[ChatMessage.from_user("Test message")],
            streaming_callback=streaming_callback
        )
        
        # Verify the API was called with the correct parameters
        _, kwargs = mock_create.call_args
        assert "stream_options" in kwargs
        assert kwargs["stream_options"]["include_usage"] is True
        
        # Verify callback was called exactly once (only for the content chunk)
        assert len(callback_chunks) == 1
        
        # Verify the result contains the expected message
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a test response"
