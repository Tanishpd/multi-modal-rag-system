# llm_qa.py
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import traceback

class LLMQA:
    def __init__(self, model_name='google/flan-t5-base'):
        print(f"Loading LLM model: {model_name}")
        self.model_name = model_name

        self.device = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU' if self.device == 0 else 'CPU'
        print(f"Target device: {device_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # create transformers pipeline (this returns list[dict] with 'generated_text')
            self.pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device=self.device,
                temperature=0.7
            )
            # keep a small wrapper for LangChain compatibility if required
            self.llm = HuggingFacePipeline(pipeline=self.pipe)

            self.prompt_template = """Based on the following context, answer the question. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

            print(f"LLM loaded on {device_name}")

        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise

    def _call_model(self, prompt):
        """
        Call the underlying transformers pipeline and return the generated text.
        Safely handle different return shapes.
        """
        try:
            gen = self.pipe(prompt, max_length=512)
            # expected: list of dicts with 'generated_text'
            if isinstance(gen, list) and len(gen) > 0:
                if isinstance(gen[0], dict) and 'generated_text' in gen[0]:
                    return gen[0]['generated_text'].strip()
                # fallback to join any strings
                return str(gen[0]).strip()
            # fallback
            return str(gen).strip()
        except Exception as e:
            print(f"Model generation error: {e}")
            traceback.print_exc()
            return None

    def generate_answer(self, query, context_chunks):
        context_text = "\n\n".join([
            f"[Source: {chunk.get('source','unknown')}]\n{chunk.get('content','')[:2000]}"
            for chunk in context_chunks[:5]
        ])

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        result_text = self._call_model(prompt)
        if not result_text:
            return "Sorry, I encountered an error generating the answer."
        return result_text

    def generate_answer_with_citations(self, query, search_results):
        context_chunks = [result['chunk'] for result in search_results]
        answer = self.generate_answer(query, context_chunks)

        citations = []
        for i, result in enumerate(search_results[:3]):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk.get('source'),
                'page': chunk.get('page'),
                'type': chunk.get('type'),
                'relevance_score': float(result.get('score', 0.0))
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(context_chunks)
        }


class SimpleQA:
    def __init__(self):
        pass

    def generate_answer_with_citations(self, query, search_results):
        if not search_results:
            return {
                'answer': "No relevant information found in the document.",
                'citations': [],
                'context_used': 0
            }
        top_chunks = search_results[:3]

        answer_parts = []
        for result in top_chunks:
            chunk = result['chunk']
            snippet = chunk['content'][:300].strip()
            if snippet:
                answer_parts.append(f"From {chunk.get('source','unknown')}: {snippet}...")

        answer = "\n\n".join(answer_parts) if answer_parts else "No relevant information found."

        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk.get('source'),
                'page': chunk.get('page'),
                'type': chunk.get('type'),
                'relevance_score': float(result.get('score', 0.0))
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(search_results)
        }


if __name__ == "__main__":
    # quick local smoke test (no model required for SimpleQA)
    test_results = [
        {
            'chunk': {
                'content': 'Qatar economy grew by 5% in 2024 driven by strong non-hydrocarbon sector growth.',
                'page': 1,
                'type': 'text',
                'source': 'Page 1'
            },
            'score': 0.85
        },
        {
            'chunk': {
                'content': 'The banking sector remains healthy with strong capital ratios.',
                'page': 2,
                'type': 'text',
                'source': 'Page 2'
            },
            'score': 0.72
        }
    ]
    qa = SimpleQA()
    print(qa.generate_answer_with_citations("What is Qatar's growth?", test_results))
