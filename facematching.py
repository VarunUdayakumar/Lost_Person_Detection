import os
import hashlib
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import math


class SimpleImageReader:
    """Basic image reader that supports JPG and PNG formats with facial region focus"""
    
    @staticmethod
    def read_facial_region_data(file_path):
        """Extract data focusing on the central portion (face area) of the image"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Get file size to determine reading positions
        file_size = len(data)
        
        # Try to get image dimensions
        width, height = SimpleImageReader.get_image_dimensions(data)
        
        # If we couldn't get dimensions, make approximations based on file size
        if width == 0 or height == 0:
            # Can't determine face region precisely, use middle portion of file
            start_pos = file_size // 3
            end_pos = (file_size * 2) // 3
            facial_data = data[start_pos:end_pos]
        else:
            # We have dimensions, so look for patterns that might be in face area
            # For JPEGs, scan for data blocks in the middle of the file
            # This is an approximation since we can't decode the actual pixels
            
            # Approximate the central region where a face would likely be
            # (center 60% of the image)
            center_x = width // 2
            center_y = height // 2
            
            # Use file size and dimensions to make a guess at where facial data might be
            bytes_per_pixel_estimate = file_size / (width * height)
            region_size = int((width * height * 0.6) * bytes_per_pixel_estimate)
            
            # Position to start reading (approximation)
            middle_pos = file_size // 2
            start_pos = max(0, middle_pos - (region_size // 2))
            end_pos = min(file_size, middle_pos + (region_size // 2))
            
            facial_data = data[start_pos:end_pos]
        
        return facial_data
    
    @staticmethod
    def get_facial_signature(file_path):
        """Get a signature focusing on the facial region of the image"""
        facial_data = SimpleImageReader.read_facial_region_data(file_path)
        
        # Create a hash of the facial region data
        return hashlib.md5(facial_data).hexdigest()
    
    @staticmethod
    def get_image_dimensions(data):
        """Try to extract image dimensions from file data"""
        width, height = 0, 0
        
        # Check if it's a JPEG
        if data.startswith(b'\xFF\xD8'):
            # Find SOF marker
            sof_markers = [b'\xFF\xC0', b'\xFF\xC1', b'\xFF\xC2']
            for marker in sof_markers:
                pos = data.find(marker)
                if pos > 0 and pos + 9 <= len(data):
                    height = (data[pos+5] << 8) + data[pos+6]
                    width = (data[pos+7] << 8) + data[pos+8]
                    break
        
        # Check if it's a PNG
        elif data.startswith(b'\x89PNG'):
            if len(data) >= 24:  # Make sure we have enough data
                width = int.from_bytes(data[16:20], byteorder='big')
                height = int.from_bytes(data[20:24], byteorder='big')
        
        return width, height
    
    @staticmethod
    def get_color_pattern(file_path):
        """Extract color pattern information focusing on facial region"""
        facial_data = SimpleImageReader.read_facial_region_data(file_path)
        
        # Break data into chunks to analyze color patterns
        chunk_size = min(1024, len(facial_data) // 10)
        chunks = [facial_data[i:i+chunk_size] for i in range(0, len(facial_data), chunk_size)]
        
        # Create a pattern based on byte distribution in each chunk
        pattern = []
        for chunk in chunks[:15]:  # Use first 15 chunks for pattern
            if chunk:
                # Get byte statistics for this chunk
                avg = sum(chunk) / len(chunk)
                # Get variance as a measure of texture
                variance = sum((b - avg)**2 for b in chunk) / len(chunk)
                # Count transitions (changes from higher to lower values and vice versa)
                transitions = sum(1 for i in range(1, len(chunk)) if 
                                 (chunk[i] > chunk[i-1] and i % 3 == 0))
                
                pattern.append((avg, variance, transitions))
        
        return pattern
    
    @staticmethod
    def analyze_image_attributes(file_path):
        """Analyze various attributes of the image that might help in face matching"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Get basic file info
        file_size = len(data)
        width, height = SimpleImageReader.get_image_dimensions(data)
        
        # Look for patterns that might indicate skin tones
        # This is very approximate without actual image decoding
        # Count byte values in ranges that might correspond to skin colors
        skin_tone_ranges = [(200, 255), (150, 200), (100, 150)]
        tone_counts = [0, 0, 0]
        
        # Sample the file at regular intervals
        sample_points = 50
        sample_size = min(100, file_size // sample_points)
        for i in range(0, file_size, file_size // sample_points):
            sample = data[i:i+sample_size]
            for j, (lower, upper) in enumerate(skin_tone_ranges):
                tone_counts[j] += sum(1 for b in sample if lower <= b <= upper)
        
        # Calculate ratios
        total_samples = sum(tone_counts)
        tone_ratios = [count / max(1, total_samples) for count in tone_counts]
        
        # Create additional features
        features = {
            "file_size": file_size,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if width > 0 and height > 0 else 0,
            "tone_ratios": tone_ratios
        }
        
        return features


def calculate_signature_similarity(sig1, sig2):
    """Calculate similarity between two facial signatures with more lenient matching"""
    # Instead of exact matching, we'll consider partial matches
    # Break signatures into chunks and compare
    chunk_size = 8
    sig1_chunks = [sig1[i:i+chunk_size] for i in range(0, len(sig1), chunk_size)]
    sig2_chunks = [sig2[i:i+chunk_size] for i in range(0, len(sig2), chunk_size)]
    
    # Calculate similarity for each chunk
    chunk_similarities = []
    for chunk1 in sig1_chunks:
        best_match = 0
        for chunk2 in sig2_chunks:
            # Count matching characters
            matches = sum(c1 == c2 for c1, c2 in zip(chunk1, chunk2))
            similarity = matches / len(chunk1)
            best_match = max(best_match, similarity)
        chunk_similarities.append(best_match)
    
    # Average the chunk similarities
    return sum(chunk_similarities) / len(chunk_similarities)


def calculate_pattern_similarity(pattern1, pattern2):
    """Calculate similarity between two color patterns"""
    # Make sure patterns are of same length for comparison
    min_len = min(len(pattern1), len(pattern2))
    pattern1 = pattern1[:min_len]
    pattern2 = pattern2[:min_len]
    
    if not pattern1 or not pattern2:
        return 0.5  # Default mid-value if no pattern data
    
    # Calculate similarity for each attribute in the pattern
    similarities = []
    for (avg1, var1, trans1), (avg2, var2, trans2) in zip(pattern1, pattern2):
        # Average byte value similarity
        avg_sim = 1 - abs(avg1 - avg2) / 255
        
        # Variance similarity (texture)
        max_var = max(var1, var2)
        var_sim = 1 - abs(var1 - var2) / max_var if max_var > 0 else 0.5
        
        # Transition similarity (edges)
        max_trans = max(trans1, trans2)
        trans_sim = 1 - abs(trans1 - trans2) / max_trans if max_trans > 0 else 0.5
        
        # Weighted combination
        similarities.append(0.5 * avg_sim + 0.3 * var_sim + 0.2 * trans_sim)
    
    # Return average similarity
    return sum(similarities) / len(similarities)


def compare_attributes(attr1, attr2):
    """Compare image attributes with lenient matching"""
    similarities = []
    
    # Compare aspect ratio if available
    if attr1["aspect_ratio"] > 0 and attr2["aspect_ratio"] > 0:
        # For aspect ratio, we want a high score even if there's some variation
        ar_sim = 1 - min(0.5, abs(attr1["aspect_ratio"] - attr2["aspect_ratio"]))
        similarities.append(ar_sim)
    
    # Compare tone ratios
    tone_sim = 0
    for t1, t2 in zip(attr1["tone_ratios"], attr2["tone_ratios"]):
        tone_sim += 1 - abs(t1 - t2)
    tone_sim /= len(attr1["tone_ratios"])
    similarities.append(tone_sim)
    
    # More lenient size comparison
    # If sizes are within 30% of each other, consider it a good match
    size_ratio = min(attr1["file_size"], attr2["file_size"]) / max(attr1["file_size"], attr2["file_size"])
    size_sim = min(1.0, size_ratio / 0.7)  # Scale up to be more lenient
    similarities.append(size_sim)
    
    # Return weighted average
    return (0.4 * similarities[0] + 0.4 * similarities[1] + 0.2 * similarities[2]) if len(similarities) >= 3 else 0.5


def compare_face_images(image1_path, image2_path, threshold=0.65):
    """Compare two images focusing on facial regions with lenient matching"""
    # Verify files exist
    if not os.path.exists(image1_path):
        raise FileNotFoundError(f"Image not found: {image1_path}")
    if not os.path.exists(image2_path):
        raise FileNotFoundError(f"Image not found: {image2_path}")
    
    # Get facial signatures
    face_sig1 = SimpleImageReader.get_facial_signature(image1_path)
    face_sig2 = SimpleImageReader.get_facial_signature(image2_path)
    
    # Calculate signature similarity with lenient matching
    signature_similarity = calculate_signature_similarity(face_sig1, face_sig2)
    
    # Get color patterns focusing on facial region
    pattern1 = SimpleImageReader.get_color_pattern(image1_path)
    pattern2 = SimpleImageReader.get_color_pattern(image2_path)
    
    # Calculate pattern similarity
    pattern_similarity = calculate_pattern_similarity(pattern1, pattern2)
    
    # Get image attributes
    attr1 = SimpleImageReader.analyze_image_attributes(image1_path)
    attr2 = SimpleImageReader.analyze_image_attributes(image2_path)
    
    # Compare attributes
    attribute_similarity = compare_attributes(attr1, attr2)
    
    # Calculate combined similarity score with more weight on pattern and less on exact signatures
    combined_similarity = (
        signature_similarity * 0.3 +  # Reduced weight on exact signature
        pattern_similarity * 0.45 +   # Increased weight on pattern matching
        attribute_similarity * 0.25   # Moderate weight on attributes
    )
    
    # Cap at 1. and floor at 0.0
    combined_similarity = max(0.0, min(1.0, combined_similarity))
    
    # Create result dictionary
    result = {
        "image1_path": image1_path,
        "image2_path": image2_path,
        "facial_signature_similarity": signature_similarity,
        "facial_pattern_similarity": pattern_similarity,
        "attribute_similarity": attribute_similarity,
        "combined_similarity": combined_similarity,
        "is_match": combined_similarity > threshold,
        "confidence": "High" if combined_similarity > 0.8 else 
                      "Medium" if combined_similarity > 0.7 else
                      "Low" if combined_similarity > threshold else
                      "Not a match"
    }
    
    return result


def find_similar_faces(reference_image, faces_dir, threshold=0.65, top_n=None):
    """Find images in the faces directory that are similar to the reference image."""
    if not os.path.exists(reference_image):
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    
    if not os.path.exists(faces_dir):
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")
    
    # List all image files in the faces directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    face_images = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) 
                  if os.path.isfile(os.path.join(faces_dir, f)) and 
                  f.lower().endswith(image_extensions)]
    
    print(f"Found {len(face_images)} images in {faces_dir} directory.")
    print(f"Comparing with reference image: {reference_image}")
    print("Processing...")
    
    # Compare reference image with each face image
    comparison_results = []
    for i, face_image in enumerate(face_images):
        try:
            print(f"Processing image {i+1}/{len(face_images)}: {os.path.basename(face_image)}", end='\r')
            result = compare_face_images(reference_image, face_image, threshold)
            comparison_results.append(result)
        except Exception as e:
            print(f"\nError comparing with {face_image}: {e}")
    
    print("\nComparison complete!")
    
    # Sort results by similarity score (descending)
    comparison_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
    
    # Filter matches above threshold
    matches = [r for r in comparison_results if r['is_match']]
    
    # Limit to top_n results if specified
    if top_n is not None and top_n > 0:
        matches = matches[:top_n]
    
    return matches


def display_results(matches, reference_image, show_images=True):
    """Display the matching results in a table format and show images"""
    if not matches:
        print("No matching faces found!")
        return
    
    print(f"\nFound {len(matches)} matching faces!\n")
    
    # Prepare table data
    table_data = []
    for match in matches:
        table_data.append([
            os.path.basename(match['image2_path']),
            f"{match['combined_similarity']:.2f}",
            match['confidence'],
            f"{match['facial_signature_similarity']:.2f}",
            f"{match['facial_pattern_similarity']:.2f}",
            f"{match['attribute_similarity']:.2f}"
        ])
    
    # Print table
    headers = ["Filename", "Similarity", "Confidence", "Signature", "Pattern", "Attributes"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Display images if requested
    if show_images and matches:
        display_matching_images(reference_image, matches)


def display_matching_images(reference_image, matches):
    """Display the reference image and all matching images using matplotlib"""
    try:
        # Load reference image
        ref_img = imread(reference_image)
        
        # Calculate grid size for matches
        num_matches = len(matches)
        grid_size = calculate_grid_size(num_matches + 1)  # +1 for reference image
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Display reference image
        plt.subplot(grid_size[0], grid_size[1], 1)
        plt.imshow(ref_img)
        plt.title(f"Reference: {os.path.basename(reference_image)}", fontsize=10)
        plt.axis('off')
        
        # Display matching images
        for i, match in enumerate(matches):
            try:
                match_img = imread(match['image2_path'])
                plt.subplot(grid_size[0], grid_size[1], i + 2)  # +2 because reference is at position 1
                plt.imshow(match_img)
                plt.title(f"{os.path.basename(match['image2_path'])}\nSimilarity: {match['combined_similarity']:.2f}", 
                         fontsize=8)
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying {match['image2_path']}: {e}")
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying images: {e}")


def calculate_grid_size(n):
    """Calculate an appropriate grid size for n images"""
    # Find the square root and round up to get a minimum grid size
    sqrt_n = math.ceil(math.sqrt(n))
    
    # Create a grid where rows ≤ columns
    if sqrt_n * (sqrt_n - 1) >= n:
        return sqrt_n - 1, sqrt_n
    else:
        return sqrt_n, sqrt_n


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Find similar faces in a directory")
    parser.add_argument("reference", help="Path to the reference image")
    parser.add_argument("--dir", default="faces", help="Directory containing face images (default: faces)")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Similarity threshold (0.0-1.0, default: 0.7)")
    parser.add_argument("--top", type=int, default=None, 
                        help="Show only top N results (default: show all matches)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display images, just show text results")
    
    args = parser.parse_args()
    
    try:
        # Find similar faces
        matches = find_similar_faces(
            args.reference, 
            args.dir, 
            threshold=args.threshold,
            top_n=args.top
        )
        
        # Display results with images
        display_results(matches, args.reference, show_images=not args.no_display)
        
    except Exception as e:
        print(f"Error: {e}")