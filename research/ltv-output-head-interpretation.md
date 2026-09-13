# What the final-state intervention changes

September 13, 2026. This is an algebraic interpretation of the frozen LTV
code, written before opening its validation scores. It changes no method,
control, or gate.

Let h be a row vector containing the final normalized state, U the
vocabulary-by-hidden output matrix, and b an optional output bias. The
unmodified logits are h U^T + b. The fitted intervention adds h X^T A,
where X and A each have 128 rows. Thus, in exact arithmetic,

$$
(h+hX^TA)U^T+b = h(U+UA^TX)^T+b.
$$

Applying the map at every step is equivalent to using a fixed output-head
update of rank at most 128. The implementation computes the activation
shift and keeps all weights frozen; it does not materialize that update.
With an identical token prefix, the backbone and its cached keys and values
are unchanged. Different chosen tokens can affect subsequent backbone
states. Prefill-only steering applies this output transformation only to
the first token decision.

The raw mean control adds a constant logit bias m U^T. For the standalone
scalar control, define a=1+c and d=m-c mean(X). Its modified state is a h+d,
so its logits can be written as

$$
a(hU^T+b)+dU^T+(1-a)b.
$$

The [frozen fit](../results/gsm8k-ltv-v1-fit.json) has c=-0.6139594253,
so a=0.3860405747 is positive. Dividing these logits by a preserves their
greedy argmax. Consequently, this scalar control is equivalent in exact
arithmetic to the original logits plus the constant bias
[dU^T+(1-a)b]/a. Its apparent dependence on h does not by itself establish
a richer decision rule than a fixed logit bias under greedy decoding.
This reduction does not apply unchanged to the norm-matched scalar,
whose rescaling depends on the current state.

These are exact-arithmetic identities. The actual BF16 cast, state
addition, and matrix multiplication can round differently from a folded
implementation. A synthetic NumPy float64 check verified both identities
and the scalar greedy choices; no replacement GPU implementation was run.

An accuracy improvement would show that a learned transformation of the
output distribution is useful. Attributing it to transferred internal
multi-step calculation would require additional evidence. These identities
apply to the final normalized state immediately before the output head;
the earlier internal-block interventions pass through additional nonlinear
layers. All existing comparisons and the planned blinded review remain
necessary.
