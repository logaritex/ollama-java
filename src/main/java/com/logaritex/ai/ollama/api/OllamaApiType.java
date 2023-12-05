/*
 * Copyright 2023-2023 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.logaritex.ai.ollama.api;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.logaritex.ai.ollama.api.OllamaApiType.ModelList.ModelResponse;

import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Java Client for the Ollama API. https://ollama.ai/
 *
 * @author Christian Tzolov
 */
public class OllamaApiType {

	/**
	 * The request object sent to the /generate endpoint.
	 *
	 * @param model (required) The model to use for completion.
	 * @param prompt (required) The prompt(s) to generate completions for.
	 * @param format (optional) The format to return the response in. Currently the only accepted value is "json".
	 * @param options (optional) additional model parameters listed in the documentation for the Modelfile such as
	 * temperature.
	 * @param system (optional) system prompt to (overrides what is defined in the Modelfile).
	 * @param template (optional) the full prompt or prompt template (overrides what is defined in the Modelfile).
	 * @param context the context parameter returned from a previous request to /generate, this can be used to keep a
	 * short conversational memory.
	 * @param stream (optional) if false the response will be returned as a single response object, rather than a stream
	 * of objects.
	 * @param raw (optional) if true no formatting will be applied to the prompt and no context will be returned. You
	 * may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API, and
	 * are managing history yourself.
	 */
	@JsonInclude(Include.NON_NULL)
	public record CompletionRequest(
			@JsonProperty("model") String model,
			@JsonProperty("prompt") String prompt,
			@JsonProperty("format") String format,
			@JsonProperty("options") Options options,
			@JsonProperty("system") String system,
			@JsonProperty("template") String template,
			@JsonProperty("context") List<Integer> context,
			@JsonProperty("stream") Boolean stream,
			@JsonProperty("raw") Boolean raw) {

		/**
		 * Short cut constructor to create a CompletionRequest without options.
		 * @param model The model used for completion.
		 * @param prompt The prompt(s) to generate completions for.
		 * @param stream Whether to stream the response.
		 */
		public CompletionRequest(String model, String prompt, Boolean stream) {
			this(model, prompt, null, null, null, null, null, stream, null);
		}

		/**
		 * Short cut constructor to create a CompletionRequest without options.
		 * @param model The model used for completion.
		 * @param prompt The prompt(s) to generate completions for.
		 * @param enableJsonFormat Whether to return the response in json format.
		 * @param stream Whether to stream the response.
		 */
		public CompletionRequest(String model, String prompt, boolean enableJsonFormat, Boolean stream) {
			this(model, prompt, (enableJsonFormat) ? "json" : null, null, null, null, null, stream, null);
		}
	}

	/**
	 * Options specified in GenerateRequest, if you add a new option here add it to the API docs also
	 */
	@JsonInclude(Include.NON_NULL)
	public record Options(
			@JsonProperty("num_keep") Integer numKeep,
			@JsonProperty("seed") Integer seed,
			@JsonProperty("num_predict") Integer numPredict,
			@JsonProperty("top_k") Integer topK,
			@JsonProperty("top_p") Float topP,
			@JsonProperty("tfs_z") Float tfsZ,
			@JsonProperty("typical_p") Float typicalP,
			@JsonProperty("repeat_last_n") Integer repeatLastN,
			@JsonProperty("temperature") Float temperature,
			@JsonProperty("repeat_penalty") Float repeatPenalty,
			@JsonProperty("presence_penalty") Float presencePenalty,
			@JsonProperty("frequency_penalty") Float frequencyPenalty,
			@JsonProperty("mirostat") Integer mirostat,
			@JsonProperty("mirostat_tau") Float mirostatTau,
			@JsonProperty("mirostat_eta") Float mirostatEta,
			@JsonProperty("penalize_newline") Boolean penalizeNewline,
			@JsonProperty("stop") String[] stop,
			Runner runner) {
	}

	/**
	 * Runner options which must be set when the model is loaded into memory
	 */
	@JsonInclude(Include.NON_NULL)
	public record Runner(
			@JsonProperty("numa") Boolean useNUMA,
			@JsonProperty("num_ctx") Integer numCtx,
			@JsonProperty("num_batch") Integer numBatch,
			@JsonProperty("num_gqa") Integer numGQA,
			@JsonProperty("num_gpu") Integer numGPU,
			@JsonProperty("main_gpu") Integer mainGPU,
			@JsonProperty("low_vram") Boolean lowVRAM,
			@JsonProperty("f16_kv") Boolean f16KV,
			@JsonProperty("logits_all") Boolean logitsAll,
			@JsonProperty("vocab_only") Boolean vocabOnly,
			@JsonProperty("use_mmap") Boolean useMMap,
			@JsonProperty("use_mlock") Boolean useMLock,
			@JsonProperty("embedding_only") Boolean embeddingOnly,
			@JsonProperty("rope_frequency_base") Float ropeFrequencyBase,
			@JsonProperty("rope_frequency_scale") Float ropeFrequencyScale,
			@JsonProperty("num_thread") Integer numThread) {
	}

	/**
	 * The response object returned from the /generate endpoint. To calculate how fast the response is generated in
	 * tokens per second (token/s), divide eval_count / eval_duration.
	 *
	 * @param model The model used for completion.
	 * @param createdAt When the request was made.
	 * @param response The completion response. Empty if the response was streamed, if not streamed, this will contain
	 * the full response
	 * @param done Whether this is the final response. If true, this response may be followed by another response with
	 * the following, additional fields: context, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration.
	 * @param context Encoding of the conversation used in this response, this can be sent in the next request to keep a
	 * conversational memory.
	 * @param totalDuration Time spent generating the response.
	 * @param loadDuration Time spent loading the model.
	 * @param promptEvalCount Number of times the prompt was evaluated.
	 * @param promptEvalDuration Time spent evaluating the prompt.
	 * @param evalCount Number of tokens in the response.
	 * @param evalDuration Time spent generating the response.
	 */
	@JsonInclude(Include.NON_NULL)
	public record GenerateResponse(
			@JsonProperty("model") String model,
			@JsonProperty("created_at") Instant createdAt,
			@JsonProperty("response") String response,
			@JsonProperty("done") Boolean done,
			@JsonProperty("context") List<Integer> context,
			@JsonProperty("total_duration") Duration totalDuration,
			@JsonProperty("load_duration") Duration loadDuration,
			@JsonProperty("prompt_eval_count") int promptEvalCount,
			@JsonProperty("prompt_eval_duration") Duration promptEvalDuration,
			@JsonProperty("eval_count") int evalCount,
			@JsonProperty("eval_duration") Duration evalDuration) {
	}

	/**
	 * Generate embeddings from a model.
	 *
	 * @param model The name of model to generate embeddings from.
	 * @param prompt The text to generate embeddings for.
	 * @param options Additional model parameters listed in the documentation for the Modelfile such as temperature.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingRequest(
			@JsonProperty("model") String model,
			@JsonProperty("prompt") String prompt,
			@JsonProperty("options") Options options) {

		/**
		 * short cut constructor to create a EmbeddingRequest without options.
		 * @param model The name of model to generate embeddings from.
		 * @param prompt The text to generate embeddings for.
		 */
		public EmbeddingRequest(String model, String prompt) {
			this(model, prompt, null);
		}
	}

	/**
	 * The response object returned from the /embedding endpoint.
	 *
	 * @param embedding The embedding generated from the model.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingResponse(
			@JsonProperty("embedding") List<Float> embedding) {
	}

	/**
	 * Modelfile file builder.
	 */
	public static class ModelFileBuilder {

		private final String from;

		private Map<String, Object> parameters;

		private String system;

		private String template;

		private String adapter;

		private String license;

		private ModelFileBuilder(String from) {
			this.from = from;
			this.parameters = new HashMap<>();
		}

		/**
		 * Defines the base model to use:
		 *
		 * <pre>{@code FROM <model name>:<tag>}</pre>
		 *
		 * @param from Base model to use. Can be a model name or a path to a model file.
		 * @return Returns this builder.
		 */
		public static ModelFileBuilder from(String from) {
			Assert.hasText(from, "from can not be null or empty.");
			return new ModelFileBuilder(from);
		}

		/**
		 * Defines a parameter that can be set when the model is run:
		 *
		 * <pre>{@code PARAMETER <parameter> <parameter value>}</pre>
		 *
		 * @param parameterName Parameter name.
		 * @param parameterValue Parameter value.
		 * @return Returns this builder.
		 */
		public ModelFileBuilder withParameter(String parameterName, Object parameterValue) {
			Assert.hasText(parameterName, "parameter can not be null or empty.");
			this.parameters.put(parameterName, parameterValue);
			return this;
		}

		/**
		 * Defines the system prompt to use when the model is run.
		 *
		 * <pre>{@code SYSTEM """<system message>"""}</pre>
		 *
		 * @param system System prompt to use when the model is run.
		 * @return Returns this builder.
		 */
		public ModelFileBuilder withSystem(String system) {
			Assert.hasText(system, "system can not be null or empty.");
			this.system = system;
			return this;
		}

		/**
		 * Defines the full prompt template to be sent to the model.
		 *
		 * <pre>{@code
		 * 	TEMPLATE """
		 *  {{- if .First }}
		 *	### System:
		 *  {{ .System }}
		 *  {{- end }}
		 *
		 *  ### User:
		 *  {{ .Prompt }}
		 *
		 *  ### Response:
		 *  """
		 * }</pre>
		 *
		 * @param template full prompt template to be sent to the model.
		 * @return Returns this builder.
		 */
		public ModelFileBuilder withTemplate(String template) {
			Assert.hasText(template, "template can not be null or empty.");
			this.template = template;
			return this;
		}

		/**
		 * Defines the (Q)LoRA adapters to apply to the model. The ADAPTER instruction specifies the LoRA adapter to
		 * apply to the base model. The value of this instruction should be an absolute path or a path relative to the
		 * Modelfile and the file must be in a GGML file format. The adapter should be tuned from the base model
		 * otherwise the behaviour is undefined.
		 *
		 * <pre>{@code ADAPTER ./ollama-lora.bin}</pre>
		 *
		 * @param adapter Adapter to use when the model is run.
		 * @return Returns this builder.
		 */
		public ModelFileBuilder withAdapter(String adapter) {
			Assert.hasText(adapter, "adapter can not be null or empty.");
			this.adapter = adapter;
			return this;
		}

		/**
		 * The LICENSE instruction allows you to specify the legal license under which the model used with this
		 * Modelfile is shared or distributed.
		 *
		 * <pre>{@code LICENSE """<license text>"""}</pre>
		 *
		 * @param license License for the model.
		 * @return Returns this builder.
		 */
		public ModelFileBuilder withLicense(String license) {
			Assert.hasText(license, "license can not be null or empty.");
			this.license = license;
			return this;
		}

		/**
		 * Builds the Modelfile.
		 * @return Returns the Modelfile as a string.
		 */
		public String build() {
			var parameterStr = !CollectionUtils.isEmpty(parameters) ? "\n" + parameters.entrySet().stream()
					.map(e -> "PARAMETER %s %s".formatted(e.getKey(), e.getValue()))
					.collect(Collectors.joining("\n")) : "";
			var systemStr = (StringUtils.hasText(system)) ? "\nSYSTEM %s".formatted(system) : "";
			var templateStr = (StringUtils.hasText(template) ? "\nTEMPLATE %s".formatted(template) : "");
			var adapterStr = (StringUtils.hasText(adapter)) ? "\nADAPTER %s".formatted(adapter) : "";
			var licenseStr = (StringUtils.hasText(license)) ? "\nLICENSE %s".formatted(license) : "";

			return "FROM %s%s%s%s%s%s".formatted(from, parameterStr, systemStr, templateStr, adapterStr, licenseStr);
		}
	}

	/**
	 * Create a model from a Modelfile.
	 * @param name The name of the model.
	 * @param modelfile The content of the Modelfile.
	 * @param stream Whether to stream the response.
	 * @param path The path to save the model to.
	 */
	@JsonInclude(Include.NON_NULL)
	public record CreateModelRequest(
			@JsonProperty("name") String name,
			@JsonProperty("modelfile") String modelfile,
			@JsonProperty("stream") Boolean stream,
			@JsonProperty("path") String path) {
	}

	/**
	 * The response object returned from the /create endpoint.
	 * @param status The status of the request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record CreateModelResponse(
			@JsonProperty("status") String status) {
	}

	/**
	 * List of model details.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ModelList(List<ModelResponse> models) {
		/**
		 * Model information response.
		 * @param name The name of the model.
		 * @param modifiedAt Last time the model was modified.
		 * @param size The content size of the model.
		 * @param digest The content digest of the model.
		 */
		public record ModelResponse(
				@JsonProperty("name") String name,
				@JsonProperty("modified_at") Instant modifiedAt,
				@JsonProperty("size") Long size,
				@JsonProperty("digest") String digest) {
		}

	}

	/**
	 * Request to show a model details.
	 * @param name The name of the model.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ShowRequest(
			@JsonProperty("name") String name) {
	}

	/**
	 * Response to show a model details.
	 * @param license The license of the model.
	 * @param modelfile The content of the Modelfile.
	 * @param parameters The parameters of the model.
	 * @param template The template of the model.
	 * @param system The system of the model.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ShowResponse(
			@JsonProperty("license") String license,
			@JsonProperty("modelfile") String modelfile,
			@JsonProperty("parameters") String parameters,
			@JsonProperty("template") String template,
			@JsonProperty("system") String system) {
	}

	/**
	 * Request to copy a model.
	 * @param source The name of the model to copy.
	 * @param destination the new name to copy the model to.
	 */
	@JsonInclude(Include.NON_NULL)
	public record CopyRequest(
			@JsonProperty("source") String source,
			@JsonProperty("destination") String destination) {
	}

	/**
	 * Request to delete a model.
	 * @param name The name of the model to delete.
	 */
	@JsonInclude(Include.NON_NULL)
	public record DeleteRequest(
			@JsonProperty("name") String name) {
	}

	/**
	 * Request to used to Pull or Push a model.
	 * @param name The name of the model to Pull or Push.
	 * @param insecure Whether to use an insecure connection.
	 * @param username The username to use for authentication.
	 * @param password The password to use for authentication.
	 */
	@JsonInclude(Include.NON_NULL)
	public record PullPushRequest(
			@JsonProperty("name") String name,
			@JsonProperty("insecure") Boolean insecure,
			@JsonProperty("username") String username,
			@JsonProperty("password") String password,
			@JsonProperty("stream") Boolean stream) {
	}

	/**
	 * Response to Pull or Push a model progress status.
	 * @param status The status of the request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ProgressResponse(
			@JsonProperty("status") String status,
			@JsonProperty("digest") String digest,
			@JsonProperty("total") Long total,
			@JsonProperty("completed") Long completed) {
	}
}
