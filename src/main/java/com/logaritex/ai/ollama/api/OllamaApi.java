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

import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.logaritex.ai.ollama.api.OllamaApiType.CompletionRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.CopyRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.CreateModelRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.CreateModelResponse;
import com.logaritex.ai.ollama.api.OllamaApiType.DeleteRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.EmbeddingRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.EmbeddingResponse;
import com.logaritex.ai.ollama.api.OllamaApiType.GenerateResponse;
import com.logaritex.ai.ollama.api.OllamaApiType.ModelFileBuilder;
import com.logaritex.ai.ollama.api.OllamaApiType.ModelList;
import com.logaritex.ai.ollama.api.OllamaApiType.ModelList.ModelResponse;
import com.logaritex.ai.ollama.api.OllamaApiType.ProgressResponse;
import com.logaritex.ai.ollama.api.OllamaApiType.PullPushRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.ShowRequest;
import com.logaritex.ai.ollama.api.OllamaApiType.ShowResponse;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.util.Assert;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StreamUtils;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Java Client for the Ollama API. https://ollama.ai/
 *
 * @author Christian Tzolov
 */
public class OllamaApi {

	private static final Log logger = LogFactory.getLog(OllamaApi.class);

	private final static String DEFAULT_BASE_URL = "http://localhost:11434";

	private final ResponseErrorHandler responseErrorHandler;
	private final RestClient restClient;
	private final WebClient webClient;

	private static class OllamaResponseErrorHandler implements ResponseErrorHandler {

		@Override
		public boolean hasError(ClientHttpResponse response) throws IOException {
			return response.getStatusCode().isError();
		}

		@Override
		public void handleError(ClientHttpResponse response) throws IOException {
			if (response.getStatusCode().isError()) {
				int statusCode = response.getStatusCode().value();
				String statusText = response.getStatusText();
				String message = StreamUtils.copyToString(response.getBody(),
						java.nio.charset.StandardCharsets.UTF_8);
				logger.warn(
						String.format("[%s] %s - %s", statusCode, statusText, message));
				throw new RuntimeException(
						String.format("[%s] %s - %s", statusCode, statusText, message));
			}
		}
	}

	/**
	 * Default constructor that uses the default localhost url.
	 */
	public OllamaApi() {
		this(DEFAULT_BASE_URL);
	}

	/**
	 * Crate a new OllamaApi instance with the given base url.
	 * @param baseUrl The base url of the Ollama server.
	 */
	public OllamaApi(String baseUrl) {
		this(baseUrl, RestClient.builder());
	}

	/**
	 * Crate a new OllamaApi instance with the given base url and {@link RestClient.Builder}.
	 * @param baseUrl The base url of the Ollama server.
	 * @param restClientBuilder The {@link RestClient.Builder} to use.
	 */
	public OllamaApi(String baseUrl, RestClient.Builder restClientBuilder) {

		this.responseErrorHandler = new OllamaResponseErrorHandler();

		Consumer<HttpHeaders> defaultHeaders = headers -> {
			headers.setContentType(MediaType.APPLICATION_JSON);
			headers.setAccept(List.of(MediaType.APPLICATION_JSON));
		};

		this.restClient = restClientBuilder
				.baseUrl(baseUrl)
				.defaultHeaders(defaultHeaders)
				.build();

		this.webClient = WebClient.builder()
				.baseUrl(baseUrl)
				.defaultHeaders(defaultHeaders)
				.build();
	}

	/**
	 * Generate a completion for the given prompt.
	 *
	 * @param completionRequest Completion request.
	 * @return Completion response.
	 */
	public GenerateResponse generate(CompletionRequest completionRequest) {
		Assert.notNull(completionRequest, "The request body can not be null.");
		Assert.isTrue(completionRequest.stream() == false, "Stream mode must be disabled.");

		return this.restClient.post()
				.uri("/api/generate")
				.body(completionRequest)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(GenerateResponse.class);
	}

	/**
	 * Generate a streaming completion for the given prompt.
	 * @param completionRequest Completion request. The request must set the stream property to true.
	 * @return Completion response as a {@link Flux} stream.
	 */
	public Flux<GenerateResponse> generateStreaming(CompletionRequest completionRequest) {
		Assert.notNull(completionRequest, "The request body can not be null.");
		Assert.isTrue(completionRequest.stream(), "Request must set the steam property to true.");

		return webClient.post()
				.uri("/api/generate")
				.body(Mono.just(completionRequest), CompletionRequest.class)
				.retrieve()
				// .onStatus(null, null)
				.bodyToFlux(GenerateResponse.class)
				.handle((data, sink) -> {
					System.out.println(data);
					sink.next(data);
				});
	}

	/**
	 * Generate embeddings from a model.
	 *
	 * @param embeddingRequest Embedding request.
	 * @return Embedding response.
	 */
	public EmbeddingResponse embedding(EmbeddingRequest embeddingRequest) {
		Assert.notNull(embeddingRequest, "The request body can not be null.");

		return this.restClient.post()
				.uri("/api/embeddings")
				.body(embeddingRequest)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(EmbeddingResponse.class);
	}

	/**
	 * Create a model from a Modelfile. It is recommended to set modelfile to the content of the Modelfile rather than
	 * just set path. This is a requirement for remote create. Remote model creation should also create any file blobs,
	 * fields such as FROM and ADAPTER, explicitly with the server using Create a Blob and the value to the path
	 * indicated in the response.
	 *
	 * @param modelName The name of the model to create.
	 * @param modelFileContent The content of the Modelfile.
	 * @return Create model response.
	 */
	public CreateModelResponse createModel(String modelName, String modelFileContent) {
		Assert.hasText(modelName, "name can not be null or empty.");
		Assert.hasText(modelFileContent, "modelFileContent can not be null or empty.");

		var createModelRequest = new CreateModelRequest(modelName, modelFileContent, false, null);

		return this.restClient.post()
				.uri("/api/create")
				.body(createModelRequest)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(CreateModelResponse.class);
	}

	/**
	 * Check if a blob exists.
	 * @param digest The digest of the blob to check.
	 * @return Returns true if the blob exists.
	 */
	public boolean isBlobExists(String digest) {
		Assert.hasText(digest, "digest can not be null or empty.");

		return this.restClient.head()
				.uri("/api/blobs/%s".formatted(digest))
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.toBodilessEntity().getStatusCode().is2xxSuccessful();
	}

	/**
	 * Create a blob with the given file digest.
	 * @param digest The digest of the blob to create.
	 * @param file The file to upload.
	 * @return Returns true if the blob was created successfully.
	 */
	public boolean createBlob(String digest, Resource file) {
		Assert.hasText(digest, "digest can not be null or empty.");

		MultiValueMap<String, Object> multipartBody = new LinkedMultiValueMap<>();
		multipartBody.add("file", file);

		return this.restClient.post()
				.uri("/api/blobs/%s".formatted(digest))
				.body(multipartBody)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.toBodilessEntity()
				.getStatusCode().is2xxSuccessful();
	}

	/**
	 * Create a model from a Modelfile. It is recommended to set modelfile to the content of the Modelfile rather than
	 * just set path. This is a requirement for remote create. Remote model creation should also create any file blobs,
	 * fields such as FROM and ADAPTER, explicitly with the server using Create a Blob and the value to the path
	 * indicated in the response.
	 *
	 * @param modelName The name of the model to create.
	 * @param modelFileContent The content of the Modelfile.
	 * @return A stream of JSON objects. When finished, status is success.
	 */
	public Flux<CreateModelResponse> createModelStreaming(String modelName, String modelFileContent) {

		Assert.hasText(modelName, "name can not be null or empty.");
		Assert.hasText(modelFileContent, "modelFileContent can not be null or empty.");

		var createModelRequest = new CreateModelRequest(modelName, modelFileContent, true, null);

		return webClient.post()
				.uri("/api/create")
				.body(Mono.just(createModelRequest), CreateModelRequest.class)
				.retrieve()
				.bodyToFlux(CreateModelResponse.class)
				.handle((data, sink) -> {
					sink.next(data);
				});
	}

	/**
	 * @return List models that are available locally.
	 */
	public List<ModelResponse> listModels() {

		ModelList modelList = this.restClient.get()
				.uri("/api/tags")
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(ModelList.class);

		return (modelList != null) ? modelList.models() : List.of();
	}

	/**
	 * Show details about a model including modelfile, template, parameters, license, and system prompt.
	 *
	 * @param model The name of the model to show.
	 * @return Show Modelfile for the given model.
	 */
	public ShowResponse showModel(String model) {
		Assert.hasText(model, "model name can not be null or empty.");

		var showRequest = new ShowRequest(model);

		return this.restClient.post()
				.uri("/api/show")
				.body(showRequest)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(ShowResponse.class);
	}

	/**
	 * Copy a model. Creates a model with another name from an existing model.
	 * @param source The name of the model to copy.
	 * @param destination The name of the new model.
	 * @return Returns true if the model was copied successfully.
	 */
	public boolean copyModel(String source, String destination) {
		Assert.hasText(source, "source model name can not be null or empty.");
		Assert.hasText(destination, "destination model name can not be null or empty.");

		var copyRequest = new CopyRequest(source, destination);

		return this.restClient.post()
				.uri("/api/copy")
				.body(copyRequest)
				.retrieve()
				.onStatus(new ResponseErrorHandler() {
					@Override
					public boolean hasError(ClientHttpResponse response) throws IOException {
						return response.getStatusCode().isError();
					}

					@Override
					public void handleError(ClientHttpResponse response) throws IOException {
						if (response.getStatusCode().isError()) {

							int statusCode = response.getStatusCode().value();
							String statusText = response.getStatusText();
							String message = StreamUtils.copyToString(response.getBody(),
									java.nio.charset.StandardCharsets.UTF_8);
							logger.warn(
									String.format("[%s] %s - %s", statusCode, statusText, message));
							// if (statusCode != 404 || (message != null && !message.contains(model))) {
							// throw new RuntimeException(
							// String.format("[%s] %s - %s", statusCode, statusText, message));
							// }
						}
					}
				})
				.toBodilessEntity()
				.getStatusCode().is2xxSuccessful();
	}

	/**
	 * Delete a model.
	 * @param model The name of the model to delete.
	 * @return Returns true if the model was deleted successfully.
	 */
	public boolean deleteModel(String model) {
		Assert.hasText(model, "model name can not be null or empty.");

		return this.restClient.method(HttpMethod.DELETE)
				.uri("/api/delete")
				.body(new DeleteRequest(model))
				.retrieve()
				.onStatus(new ResponseErrorHandler() {
					@Override
					public boolean hasError(ClientHttpResponse response) throws IOException {
						return response.getStatusCode().isError();
					}

					@Override
					public void handleError(ClientHttpResponse response) throws IOException {
						if (response.getStatusCode().isError()) {

							int statusCode = response.getStatusCode().value();
							String statusText = response.getStatusText();
							String message = StreamUtils.copyToString(response.getBody(),
									java.nio.charset.StandardCharsets.UTF_8);
							logger.warn(
									String.format("[%s] %s - %s", statusCode, statusText, message));
							if (statusCode != 404 || (message != null && !message.contains(model))) {
								throw new RuntimeException(
										String.format("[%s] %s - %s", statusCode, statusText, message));
							}
						}
					}
				})
				.toBodilessEntity()
				.getStatusCode().is2xxSuccessful();
	}

	/**
	 * Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple
	 * calls will share the same download progress. Insecure mode is enabled by default.
	 *
	 * @param model The name of the model to pull.
	 * @return Stream of ProgressResponse objects is returned. Then there is a series of downloading responses. Until
	 * any of the download is completed, the completed key may not be included. The number of files to be downloaded
	 * depends on the number of layers specified in the manifest.
	 */
	public Flux<ProgressResponse> pullModelStream(String model) {
		return pullModelStream(model, true, null, null);
	}

	/**
	 * Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple
	 * calls will share the same download progress.
	 *
	 * @param model The name of the model to pull.
	 * @param insecure Whether to skip TLS verification. allow insecure connections to the library. Only use this if you
	 * are pulling from your own library during development.
	 * @param username The username to use for authentication.
	 * @param password The password to use for authentication.
	 * @return Stream of ProgressResponse objects is returned.
	 */
	public Flux<ProgressResponse> pullModelStream(String model, boolean insecure, String username, String password) {
		return internalPullPushModelStream("/api/pull", model, insecure, username, password);
	}

	/**
	 * Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.
	 *
	 * @param model The name of the model to push in the form of 'namespace/model:tag'.
	 * @param insecure Whether to skip TLS verification. allow insecure connections to the library.
	 * @param username The username to use for authentication.
	 * @param password The password to use for authentication.
	 * @return Stream of ProgressResponse objects is returned.
	 */
	public Flux<ProgressResponse> pushModelStream(String model, boolean insecure, String username, String password) {
		return internalPullPushModelStream("/api/push", model, insecure, username, password);
	}

	private Flux<ProgressResponse> internalPullPushModelStream(String uri, String model, boolean insecure,
			String username,
			String password) {
		Assert.hasText(uri, "uri can not be null or empty.");
		Assert.hasText(model, "model name can not be null or empty.");

		var pullPushRequest = new PullPushRequest(model, insecure, username, password, true);

		return this.webClient.post()
				.uri(uri)
				.body(Mono.just(pullPushRequest), PullPushRequest.class)
				.retrieve()
				.bodyToFlux(ProgressResponse.class)
				.handle((data, sink) -> {
					sink.next(data);
				});
	}

	/**
	 * Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple
	 * calls will share the same download progress.
	 *
	 * @param model The name of the model to pull.
	 * @param insecure Whether to skip TLS verification. allow insecure connections to the library. Only use this if you
	 * are pulling from your own library during development.
	 * @param username The username to use for authentication.
	 * @param password The password to use for authentication.
	 * @return Stream of ProgressResponse objects is returned.
	 */
	public ProgressResponse pullModel(String model, boolean insecure, String username, String password) {
		return internalPullPushModel("/api/pull", model, insecure, username, password);
	}

	/**
	 * Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.
	 *
	 * @param model The name of the model to push in the form of
	 *
	 * <pre>
	 * namespace>/model:tag
	 * </pre>
	 *
	 * @param insecure Whether to skip TLS verification. allow insecure connections to the library.
	 * @param username The username to use for authentication.
	 * @param password The password to use for authentication.
	 * @return Stream of ProgressResponse objects is returned.
	 */
	public ProgressResponse pushModel(String model, boolean insecure, String username, String password) {
		return internalPullPushModel("/api/push", model, insecure, username, password);
	}

	private ProgressResponse internalPullPushModel(String uri, String model, boolean insecure, String username,
			String password) {
		Assert.hasText(uri, "uri can not be null or empty.");
		Assert.hasText(model, "model name can not be null or empty.");

		var pullPushRequest = new PullPushRequest(model, insecure, username, password, false);

		return this.restClient.post()
				.uri(uri)
				.body(pullPushRequest)
				.retrieve()
				.onStatus(this.responseErrorHandler)
				.body(ProgressResponse.class);
	}

	public static void main(String[] args) throws JsonProcessingException {

		var client = new OllamaApi();

		// var response = client.generate(new CompletionRequest("orca2",
		// "Why is the sky blue? Answer in one sentence only. no longer than 20 words. Respond using JSON", true,
		// false));

		// System.out.println(response);
		// var flux = client.generateStreaming(new CompletionRequest("orca2",
		// "Why is the sky blue? Answer in one sentence only. no longer than 20 words.", false,
		// true));

		// flux.subscribe(
		// content -> {
		// // System.out.println("role: " + choice.delta().getRole());
		// System.out.println("content: " + content.response());
		// },
		// error -> System.out.println("Error receiving: " + error),
		// () -> System.out.println("Completed!!!"));

		// var b = flux.buffer();

		// System.out.println(client.embedding(new EmbeddingRequest("orca2", "Hello World")).embedding().size());

		var b = ModelFileBuilder.from("nous-hermes")
				.withSystem("You are a content marketer who needs to come up with a short but succinct tweet. "
						+ "Make sure to include the appropriate hashtags and links. Sometimes when appropriate, describe a meme that can be included as well. "
						+ "All answers should be in the form of a tweet which has a max size of 280 characters. "
						+ "Every instruction will be the topic to create a tweet about.")
				.withParameter("temperature", 1)
				.withTemplate("""
						{{- if .First }}
						### System:
						{{ .System }}
						{{- end }}

						### User:
						{{ .Prompt }}

						### Response:
						""");

		// var modelCreateFlux = client.createModelStreaming("mario", b.build());

		// modelCreateFlux.subscribe(
		// content -> {
		// // System.out.println("role: " + choice.delta().getRole());
		// System.out.println("content: " + content.status());
		// },
		// error -> System.out.println("Error receiving: " + error),
		// () -> System.out.println("Completed!!!"));
		// modelCreateFlux.blockLast();

		System.out.println(client.listModels());

		// System.out.println(client.showModel("orca2"));

		System.out.println(
				client.isBlobExists("sha256:508ec557716d736bb1b949c4acdd5e990c1f1d3dbb70edc9474674346e9b65c3"));

		System.out.println(client.copyModel("orca-mini", "orca-mini-1"));
		// System.out.println(client.deleteModel("orca-mini"));
		System.out.println(client.listModels());

		System.out.println(client.pushModel("library/orca-mini-1:latest", true, null, null));

		// var pull = client.pullModel("orca-mini", true, null, null);
		// System.out.println(pull);

		// fux.subscribe(
		// content -> {
		// System.out.println("content: " + content);
		// },
		// error -> System.out.println("Error receiving: " + error),
		// () -> System.out.println("Completed!!!"));

		// fux.blockLast();
	}

	// curl -X POST http://localhost:11434/api/generate -d '{"model": "orca2","prompt": "Why is the sky blue?"}'
}
