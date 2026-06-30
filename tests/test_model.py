"""
Test script for Step 5: U-Net + ResNet-34 Model Architecture.
Validates model construction, forward pass, output shapes, gradient flow,
weight initialization, skip connections, and integration with real DataLoader.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from src.models.unet import UNetResNet34, BasicBlock, ResNet34Encoder, DecoderBlock, build_model
from src.data.dataset import get_dataloaders
from configs.config import (
    IN_CHANNELS, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE,
    PREPROCESSED_DIR, DROPOUT_P
)


def test_encoder_shapes():
    """Test that the ResNet-34 encoder produces correct feature map shapes."""
    print("TEST 1: Encoder feature map shapes...")
    
    encoder = ResNet34Encoder(in_channels=IN_CHANNELS)
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    x0, x1, x2, x3, x4 = encoder(x)
    
    expected = {
        'x0': (2, 64, 128, 128),
        'x1': (2, 64, 64, 64),
        'x2': (2, 128, 32, 32),
        'x3': (2, 256, 16, 16),
        'x4': (2, 512, 8, 8),
    }
    
    actual = {
        'x0': tuple(x0.shape),
        'x1': tuple(x1.shape),
        'x2': tuple(x2.shape),
        'x3': tuple(x3.shape),
        'x4': tuple(x4.shape),
    }
    
    for name in expected:
        assert actual[name] == expected[name], \
            f"{name}: {actual[name]} != {expected[name]}"
        print(f"  {name}: {actual[name]}")
    
    print("  PASSED")


def test_encoder_block_counts():
    """Verify ResNet-34 has exactly [3, 4, 6, 3] basic blocks."""
    print("\nTEST 2: Encoder block counts [3, 4, 6, 3]...")
    
    encoder = ResNet34Encoder(in_channels=IN_CHANNELS)
    
    expected_blocks = {'layer1': 3, 'layer2': 4, 'layer3': 6, 'layer4': 3}
    
    for layer_name, expected_count in expected_blocks.items():
        layer = getattr(encoder, layer_name)
        actual_count = len(layer)
        assert actual_count == expected_count, \
            f"{layer_name}: {actual_count} blocks != {expected_count}"
        
        # Verify each block is a BasicBlock
        for i, block in enumerate(layer):
            assert isinstance(block, BasicBlock), \
                f"{layer_name}[{i}] is {type(block)}, not BasicBlock"
        
        print(f"  {layer_name}: {actual_count} BasicBlocks")
    
    total = sum(expected_blocks.values())
    print(f"  Total: {total} residual blocks (matches ResNet-34)")
    print("  PASSED")


def test_decoder_block():
    """Test a single decoder block with spatial dim check."""
    print("\nTEST 3: Decoder block...")
    
    block = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
    x = torch.randn(2, 512, 8, 8)
    skip = torch.randn(2, 256, 16, 16)
    
    out = block(x, skip)
    assert out.shape == (2, 256, 16, 16), f"Got {out.shape}"
    print(f"  Up(512, 8x8) + skip(256, 16x16) -> {tuple(out.shape)}")
    
    # Test with odd spatial dimensions (should handle mismatch)
    x_odd = torch.randn(2, 128, 7, 7)
    skip_odd = torch.randn(2, 64, 15, 15)
    block_odd = DecoderBlock(128, 64, 64)
    out_odd = block_odd(x_odd, skip_odd)
    assert out_odd.shape == (2, 64, 15, 15), f"Odd dim: {out_odd.shape}"
    print(f"  Up(128, 7x7) + skip(64, 15x15) -> {tuple(out_odd.shape)} (odd dims handled)")
    
    print("  PASSED")


def test_full_model_forward():
    """Test complete forward pass with correct input/output shapes."""
    print("\nTEST 4: Full model forward pass...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                        dropout_p=DROPOUT_P)
    model.eval()
    
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    with torch.no_grad():
        out = model(x)
    
    expected_shape = (2, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
    assert out.shape == expected_shape, f"Output {out.shape} != {expected_shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("  PASSED")


def test_weight_initialization():
    """Verify Kaiming Normal init was applied correctly."""
    print("\nTEST 5: Weight initialization (Kaiming Normal)...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    
    conv_count = 0
    bn_count = 0
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_count += 1
            # Kaiming normal: mean ≈ 0, std depends on fan_out
            w = m.weight.data
            assert w.mean().abs() < 0.1, f"{name}: Conv2d mean {w.mean():.4f} too far from 0"
            assert w.std() > 0, f"{name}: Conv2d std is 0 (not initialized)"
            if m.bias is not None:
                assert m.bias.data.abs().max() < 1e-6, f"{name}: Conv2d bias not zero"
        elif isinstance(m, nn.BatchNorm2d):
            bn_count += 1
            assert (m.weight.data == 1.0).all(), f"{name}: BN weight not 1"
            assert (m.bias.data == 0.0).all(), f"{name}: BN bias not 0"
    
    print(f"  Verified {conv_count} Conv2d layers (Kaiming Normal)")
    print(f"  Verified {bn_count} BatchNorm2d layers (weight=1, bias=0)")
    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow through every layer of the model."""
    print("\nTEST 6: Gradient flow (all 134 parameter groups)...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    target = torch.randint(0, NUM_CLASSES, (2, PATCH_SIZE, PATCH_SIZE))
    
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    
    total_params = 0
    params_with_grad = 0
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1
        else:
            zero_grad_params.append(name)
    
    if zero_grad_params:
        print(f"  WARNING: No gradient in: {zero_grad_params}")
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients: {params_with_grad}/{total_params} parameter groups")
    assert params_with_grad == total_params, \
        f"Only {params_with_grad}/{total_params} have gradients"
    print("  PASSED")


def test_skip_connections():
    """Verify skip connections actually influence the output."""
    print("\nTEST 7: Skip connections influence output...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    x = torch.randn(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    with torch.no_grad():
        # Normal forward pass
        out_normal = model(x).clone()
        
        # Corrupt one encoder skip and check output changes
        x0, x1, x2, x3, x4 = model.encoder(x)
        
        # Zero out the x2 skip connection
        x2_zeroed = torch.zeros_like(x2)
        
        d4 = model.decoder4(x4, x3)
        d3_normal = model.decoder3(d4, x2)
        d3_zeroed = model.decoder3(d4, x2_zeroed)
        
        # d3 should differ when skip is zeroed
        assert not torch.equal(d3_normal, d3_zeroed), \
            "Zeroing skip connection had no effect — skip not used!"
    
    diff = (d3_normal - d3_zeroed).abs().mean().item()
    print(f"  Mean absolute diff when x2 skip zeroed: {diff:.6f}")
    assert diff > 1e-6, "Skip connection has negligible influence"
    print("  PASSED")


def test_parameter_count():
    """Test model parameter count matches expected ResNet-34 U-Net."""
    print("\nTEST 8: Parameter count...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count per component
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = total - enc_params
    
    print(f"  Encoder:  {enc_params:>12,} params")
    print(f"  Decoder:  {dec_params:>12,} params")
    print(f"  Total:    {total:>12,} params")
    print(f"  Size:     ~{total * 4 / 1024 / 1024:.1f} MB (float32)")
    
    assert 20_000_000 < total < 35_000_000, \
        f"Param count {total:,} outside expected range"
    assert total == trainable, "Some parameters are frozen unexpectedly"
    print("  PASSED")


def test_with_real_dataloader():
    """Test model with real preprocessed satellite data."""
    print("\nTEST 9: Forward pass with real satellite data...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['val']))
    
    print(f"  Input:  {tuple(images.shape)}, dtype={images.dtype}")
    print(f"  Target: {tuple(masks.shape)}, dtype={masks.dtype}")
    
    with torch.no_grad():
        logits = model(images)
    
    assert logits.shape == (4, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
    assert torch.isfinite(logits).all(), "NaN/Inf on real data"
    
    preds = torch.argmax(logits, dim=1)
    assert preds.shape == masks.shape
    
    # Softmax probabilities should sum to 1
    probs = torch.softmax(logits, dim=1)
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Softmax probabilities don't sum to 1"
    
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Predictions: unique={torch.unique(preds).tolist()}")
    print(f"  Softmax sums to 1.0: verified")
    print("  PASSED")


def test_training_step():
    """Full training step: forward, loss, backward, optimizer step."""
    print("\nTEST 10: Full training step on real data...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['train']))
    
    # Save initial weights
    w_before = model.final_conv.weight.data.clone()
    
    t0 = time.time()
    logits = model(images)
    loss = criterion(logits, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    
    # Verify weights actually changed
    w_after = model.final_conv.weight.data
    assert not torch.equal(w_before, w_after), "Weights didn't update after optimizer step"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Step time: {elapsed:.2f}s (CPU)")
    print(f"  Weights updated: yes")
    assert loss.item() > 0
    assert torch.isfinite(torch.tensor(loss.item()))
    print("  PASSED")


def test_loss_decreases():
    """Verify loss decreases over a few training steps (model can learn)."""
    print("\nTEST 11: Loss decreases over 5 steps (learnability)...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Use a fixed small batch for overfitting test
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['train']))
    
    losses = []
    for step in range(5):
        logits = model(images)
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step+1}: loss={loss.item():.4f}")
    
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("  PASSED")


def test_different_batch_sizes():
    """Test model works with various batch sizes."""
    print("\nTEST 12: Different batch sizes...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    for bs in [1, 2, 4, 8]:
        x = torch.randn(bs, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
        print(f"  Batch size {bs}: {tuple(out.shape)}")
    
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  DeforestNet — Step 5: Model Architecture Tests (Extended)")
    print("=" * 60)
    
    test_encoder_shapes()
    test_encoder_block_counts()
    test_decoder_block()
    test_full_model_forward()
    test_weight_initialization()
    test_gradient_flow()
    test_skip_connections()
    test_parameter_count()
    test_with_real_dataloader()
    test_training_step()
    test_loss_decreases()
    test_different_batch_sizes()
    
    print("\n" + "=" * 60)
    print("  ALL 12 TESTS PASSED!")
    print("=" * 60)
