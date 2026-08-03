# Design decisions

1. **Two-step composition — RESOLVED.** Use `product`: `p(CA) = p(abnormal) × p(CA | abnormal)`. Confirmed by the project owner on 2026-08-02.
2. **Averaging order — RESOLVED.** Compose each image's two probabilities first, then average the composed CA score within encounter. This follows the paper's image-to-encounter reporting rule without multiplying averages from different images.
3. **Diffusion channels — RESOLVED.** Use one channel because the registered inputs and CompactCNN contract are grayscale; 224-pixel RGB replication is confined to optional view inference.
4. **Duke generator scope — RESOLVED.** Train one three-class CA/LVH/GEN conditional model per task, and sample the CA class for primary augmentation. This follows paper lines 176-177 and keeps class conditioning testable.
5. **Maximum synthetic count — RESOLVED.** Search through 3,000 in increments of 100 and report the validation-selected count. Test data cannot alter this choice.
6. **Registration reference — RESOLVED.** Cache the component-wise median fitted circular geometry over training images only. Validation, test, and synthetic inputs reuse it unchanged.
