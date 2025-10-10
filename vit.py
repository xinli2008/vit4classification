"""
A PyTorch Implementation of Vision Transformer (ViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    
    def __init__(self, image_size: int, patch_size: int, num_channels: int, hidden_size: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
        Returns:
            patch_embeddings: [batch_size, num_patches, hidden_size]
        """
        # Project patches: [B, C, H, W] -> [B, hidden_size, H//P, W//P]
        patch_embeddings = self.projection(pixel_values)
        
        # Flatten and transpose: [B, hidden_size, H//P, W//P] -> [B, num_patches, hidden_size]
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        
        return patch_embeddings


class ViTEmbeddings(nn.Module):
    """Construct embeddings from patch, position and cls token embeddings"""
    
    def __init__(self, image_size: int, patch_size: int, num_channels: int, 
                 hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embeddings = PatchEmbedding(image_size, patch_size, num_channels, hidden_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Position embeddings for [CLS] + patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size)
        )
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
        Returns:
            embeddings: [batch_size, seq_len, hidden_size] where seq_len = num_patches + 1
        """
        batch_size = pixel_values.shape[0]
        
        # Get patch embeddings
        patch_embeddings = self.patch_embeddings(pixel_values)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with patch embeddings
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        assert hidden_size % num_attention_heads == 0, \
            f"Hidden size {hidden_size} must be divisible by number of attention heads {num_attention_heads}"
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=False)
        
        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            attention_output: [batch_size, seq_len, hidden_size]
        """
        # Generate Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape and project
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        attention_output = self.dense(context_layer)
        
        return attention_output


class MLP(nn.Module):
    """Multi-Layer Perceptron (Feed Forward Network)"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and MLP"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int,
                 hidden_dropout_prob: float, attention_probs_dropout_prob: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states
        
        # MLP with residual connection
        mlp_output = self.mlp(self.layernorm_after(hidden_states))
        hidden_states = mlp_output + hidden_states
        
        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, num_hidden_layers: int,
                 intermediate_size: int, hidden_dropout_prob: float, attention_probs_dropout_prob: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob
            ) for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ViTClassificationHead(nn.Module):
    """Classification head for Vision Transformer"""
    
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            logits: [batch_size, num_labels]
        """
        # Extract CLS token representation
        cls_token_state = hidden_states[:, 0]
        
        # Apply layer normalization and classification
        cls_token_state = self.layernorm(cls_token_state)
        logits = self.classifier(cls_token_state)
        
        return logits


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification"""
    
    def __init__(self, 
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 1,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 num_labels: int = 10):
        super().__init__()
        
        # Validate parameters
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        assert hidden_size % num_attention_heads == 0, \
            f"Hidden size {hidden_size} must be divisible by number of attention heads {num_attention_heads}"
        
        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_labels = num_labels
        
        # Calculate derived parameters
        self.num_patches = (image_size // patch_size) ** 2
        self.head_dim = hidden_size // num_attention_heads
        
        # Core components
        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob
        )
        
        self.encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        
        self.classifier = ViTClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
        Returns:
            logits: [batch_size, num_labels]
        """
        # Get embeddings
        embedding_output = self.embeddings(pixel_values)
        
        # Pass through transformer encoder
        encoder_output = self.encoder(embedding_output)
        
        # Get classification logits
        logits = self.classifier(encoder_output)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory functions for common ViT configurations
def vit_tiny(num_channels: int = 1, num_labels: int = 10):
    """ViT-Tiny: Small model for experimentation"""
    model = VisionTransformer(
        image_size=224, patch_size=16, num_channels=num_channels,
        hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
        intermediate_size=768, num_labels=num_labels
    )
    return model


def vit_small(num_channels: int = 1, num_labels: int = 10):
    """ViT-Small: Moderate sized model"""
    model = VisionTransformer(
        image_size=224, patch_size=16, num_channels=num_channels,
        hidden_size=384, num_attention_heads=6, num_hidden_layers=12,
        intermediate_size=1536, num_labels=num_labels
    )
    return model


def vit_base(num_channels: int = 1, num_labels: int = 10):
    """ViT-Base: Standard model similar to original paper"""
    model = VisionTransformer(
        image_size=224, patch_size=16, num_channels=num_channels,
        hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
        intermediate_size=3072, num_labels=num_labels
    )
    return model


def vit_large(num_channels: int = 1, num_labels: int = 10):
    """ViT-Large: Large model for high-accuracy tasks"""
    model = VisionTransformer(
        image_size=224, patch_size=16, num_channels=num_channels,
        hidden_size=1024, num_attention_heads=16, num_hidden_layers=24,
        intermediate_size=4096, num_labels=num_labels
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Vision Transformer...")
    
    # Create model with custom parameters
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_channels=1,  # MNIST
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,  # Smaller for testing
        intermediate_size=3072,
        num_labels=10
    )
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 224, 224)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output logits (first sample): {output[0]}")
    
    # Test factory functions
    print("\nTesting factory functions...")
    model_tiny = vit_tiny()
    model_small = vit_small()
    model_base = vit_base()
    model_large = vit_large()
    
    print(f"ViT-Tiny parameters: {model_tiny.get_num_params():,}")
    print(f"ViT-Small parameters: {model_small.get_num_params():,}")
    print(f"ViT-Base parameters: {model_base.get_num_params():,}")
    print(f"ViT-Large parameters: {model_large.get_num_params():,}")
    
    print("All tests passed!")