export function DisclaimerBanner() {
  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2">
      <p className="text-xs text-amber-800 text-center leading-relaxed">
        <strong>Decision Support Tool — Not a Regulatory Determination.</strong>{' '}
        PharmaGuard AI uses statistical heuristics on a synthetic demonstration dataset.
        Results do not constitute FDA approval readiness, do not establish drug causality,
        and require review by a qualified pharmacovigilance and regulatory affairs professional
        before any action is taken.
      </p>
    </div>
  )
}
