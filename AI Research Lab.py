#!/usr/bin/env python3
"""
Enhanced Interactive Neural Network NLP Simulator
with CSV Word Database & Multi-threading Support

Author: AI Research Lab
Version: 2.0
License: Educational Use
"""

import sys
import time
import json
import csv
import numpy as np
import random
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import deque, defaultdict
from datetime import datetime
import os

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ============================================================================
# CORE NEURAL NETWORK ENGINE (Same as before)
# ============================================================================

class NeuralNetwork:
    """Feed-Forward Neural Network with real weights and activations"""
    
    def __init__(self, input_size=10, hidden_layers=[8, 6, 4], output_size=6):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activations = []
        self.initialize_weights()
        
        # Performance tracking
        self.processing_time = 0
        self.activation_history = []
        
    def initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            # Weight matrix: rows = current layer, cols = next layer
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            weight_matrix = np.random.uniform(-limit, limit, 
                                             (self.layer_sizes[i], self.layer_sizes[i + 1]))
            bias_vector = np.zeros(self.layer_sizes[i + 1])
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
    def forward_propagate(self, input_vector):
        """Execute forward propagation with activation tracking"""
        start_time = time.time()
        self.activations = [input_vector.copy()]
        
        current_activation = input_vector
        layer_activations = []
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            z = np.dot(current_activation, weight) + bias
            
            # Activation function (Leaky ReLU for hidden, softmax for output)
            if i == len(self.weights) - 1:  # Output layer
                # Softmax
                exp_z = np.exp(z - np.max(z))
                current_activation = exp_z / exp_z.sum()
            else:  # Hidden layers
                # Leaky ReLU
                current_activation = np.maximum(0.01 * z, z)
            
            self.activations.append(current_activation.copy())
            layer_activations.append({
                'layer': i,
                'z_values': z.tolist(),
                'activations': current_activation.tolist(),
                'max_activation': float(np.max(current_activation)),
                'avg_activation': float(np.mean(current_activation))
            })
        
        self.processing_time = (time.time() - start_time) * 1000  # ms
        self.activation_history = layer_activations
        
        return current_activation, layer_activations
    
    def get_weight_statistics(self):
        """Get statistics about all weights"""
        all_weights = np.concatenate([w.flatten() for w in self.weights])
        return {
            'total_weights': len(all_weights),
            'mean_weight': float(np.mean(all_weights)),
            'std_weight': float(np.std(all_weights)),
            'max_weight': float(np.max(all_weights)),
            'min_weight': float(np.min(all_weights)),
            'positive_weights': int(np.sum(all_weights > 0)),
            'negative_weights': int(np.sum(all_weights < 0)),
            'zero_weights': int(np.sum(np.abs(all_weights) < 1e-10))
        }

# ============================================================================
# ORIGINAL NLP PROCESSOR & SENTENCE GENERATOR
# ============================================================================

class NLPProcessor:
    """Original NLP Processor (for backward compatibility)"""
    
    # Vocabulary and word categories
    WORD_CATEGORIES = {
        'verbs': ['learn', 'create', 'build', 'think', 'solve', 'explore', 
                 'understand', 'design', 'code', 'analyze'],
        'nouns': ['idea', 'system', 'solution', 'model', 'network', 
                 'algorithm', 'data', 'concept', 'project', 'theory'],
        'adjectives': ['smart', 'efficient', 'complex', 'simple', 'innovative',
                      'logical', 'dynamic', 'adaptive', 'robust', 'scalable']
    }
    
    # Sentence templates
    SENTENCE_TEMPLATES = {
        'basic': {
            'past': "Yesterday, I {verb}ed a {adjective} {noun}.",
            'present': "I {verb} a {adjective} {noun}.",
            'future': "Tomorrow, I will {verb} a {adjective} {noun}."
        },
        'medium': {
            'past': "The team successfully {verb}ed an innovative {noun} that was {adjective}.",
            'present': "We are {verb}ing a {adjective} {noun} system for our project.",
            'future': "We plan to {verb} a more {adjective} {noun} in the coming weeks."
        },
        'advanced': {
            'past': "Through extensive research, we had {verb}ed a highly {adjective} {noun} architecture.",
            'present': "Our research is currently {verb}ing a novel {adjective} {noun} methodology.",
            'future': "Future developments will {verb} an even more {adjective} {noun} framework."
        }
    }
    
    def __init__(self):
        self.word_embeddings = self.create_word_embeddings()
        
    def create_word_embeddings(self):
        """Create simple word embeddings for demo"""
        embeddings = {}
        vocab_size = 10
        
        # Create embeddings for all words
        all_words = []
        for category, words in self.WORD_CATEGORIES.items():
            all_words.extend(words)
        
        for i, word in enumerate(all_words):
            # Simple one-hot like encoding with some semantic features
            embedding = np.zeros(vocab_size)
            embedding[i % vocab_size] = 1.0
            # Add some category information
            if word in self.WORD_CATEGORIES['verbs']:
                embedding[0] = 0.8
            elif word in self.WORD_CATEGORIES['nouns']:
                embedding[1] = 0.8
            elif word in self.WORD_CATEGORIES['adjectives']:
                embedding[2] = 0.8
            embeddings[word] = embedding
        
        return embeddings
    
    def process_word(self, word):
        """Process input word and return embedding"""
        word_lower = word.lower().strip()
        
        if word_lower in self.word_embeddings:
            return self.word_embeddings[word_lower]
        else:
            # Generate random embedding for unknown words
            embedding = np.random.randn(10)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def detect_word_type(self, word):
        """Detect if word is verb, noun, or adjective"""
        word_lower = word.lower().strip()
        for word_type, words in self.WORD_CATEGORIES.items():
            if word_lower in words:
                return word_type[:-1]  # Remove 's' from plural
        return "unknown"
    
    def generate_sentences(self, word, nn_output):
        """Generate sentences based on neural network output"""
        word_type = self.detect_word_type(word)
        
        # Use NN output to select sentence variations
        complexity_idx = np.argmax(nn_output[:3])
        tense_idx = np.argmax(nn_output[3:])
        
        complexities = ['basic', 'medium', 'advanced']
        tenses = ['past', 'present', 'future']
        
        selected_complexity = complexities[complexity_idx]
        selected_tense = tenses[tense_idx]
        
        # Get template
        template = self.SENTENCE_TEMPLATES[selected_complexity][selected_tense]
        
        # Prepare word variations
        verb_form = word
        if selected_tense == 'past' and word.endswith('e'):
            verb_form = word + 'd'
        elif selected_tense == 'past':
            verb_form = word + 'ed'
        elif selected_tense == 'present':
            if not word.endswith('ing'):
                verb_form = word + 'ing'
        
        # Generate sentence
        sentence = template.format(
            verb=verb_form,
            adjective=self.get_similar_adjective(word),
            noun=self.get_similar_noun(word)
        )
        
        # Generate alternatives
        all_sentences = {}
        for complexity in complexities:
            all_sentences[complexity] = {}
            for tense in tenses:
                template = self.SENTENCE_TEMPLATES[complexity][tense]
                all_sentences[complexity][tense] = template.format(
                    verb=verb_form,
                    adjective=self.get_similar_adjective(word),
                    noun=self.get_similar_noun(word)
                )
        
        return {
            'primary_sentence': sentence,
            'all_sentences': all_sentences,
            'complexity': selected_complexity,
            'tense': selected_tense,
            'word_type': word_type,
            'confidence': float(np.max(nn_output))
        }
    
    def get_similar_adjective(self, word):
        """Get a related adjective"""
        adjectives = self.WORD_CATEGORIES['adjectives']
        # Simple hash-based selection for consistency
        idx = hash(word) % len(adjectives)
        return adjectives[idx]
    
    def get_similar_noun(self, word):
        """Get a related noun"""
        nouns = self.WORD_CATEGORIES['nouns']
        idx = hash(word) % len(nouns)
        return nouns[idx]

# ============================================================================
# CSV WORD DATABASE MANAGER
# ============================================================================

class WordDatabase:
    """Manages CSV-based word database with parts of speech"""
    
    def __init__(self, csv_path="word_database.csv"):
        self.csv_path = csv_path
        self.word_data = {}  # word -> data dict
        self.pos_index = defaultdict(list)  # POS -> list of words
        self.synonyms = defaultdict(list)  # word -> synonyms
        self.sentence_templates = defaultdict(list)  # POS -> templates
        
        # Parts of speech categories
        self.PARTS_OF_SPEECH = {
            'n': 'Noun',
            'v': 'Verb',
            'adj': 'Adjective',
            'adv': 'Adverb',
            'prep': 'Preposition',
            'conj': 'Conjunction',
            'pron': 'Pronoun',
            'det': 'Determiner'
        }
        
        # Extended sentence templates for all POS
        self.init_sentence_templates()
        
    def init_sentence_templates(self):
        """Initialize sentence templates for each part of speech"""
        
        # Noun templates
        self.sentence_templates['n'] = [
            ("The {word} is very {adj}.", "basic"),
            ("I saw a {adj} {word} in the {n2}.", "medium"),
            ("The {adj} {word} that we found was truly {adj2}.", "advanced"),
            ("My favorite {word} is the one that's {adj}.", "basic"),
            ("After careful consideration, the {word} proved to be {adj}.", "advanced"),
            ("We need to buy a new {word} for the {n2}.", "medium"),
            ("The {word}'s design is remarkably {adj}.", "medium"),
            ("Every {word} has its own unique {n2}.", "basic")
        ]
        
        # Verb templates
        self.sentence_templates['v'] = [
            ("I {word} every day.", "basic"),
            ("We should {word} the {n} carefully.", "medium"),
            ("To {word} effectively requires significant {n}.", "advanced"),
            ("She will {word} the {n} tomorrow.", "basic"),
            ("The team plans to {word} the entire {n}.", "medium"),
            ("After much deliberation, we decided to {word} the {adj} {n}.", "advanced"),
            ("You can {word} it if you try.", "basic"),
            ("The process of {word}ing requires patience.", "medium")
        ]
        
        # Adjective templates
        self.sentence_templates['adj'] = [
            ("The {n} is very {word}.", "basic"),
            ("It's a {word} {n} with amazing features.", "medium"),
            ("The {adj} {n} was surprisingly {word}.", "advanced"),
            ("I find it {word} and interesting.", "basic"),
            ("This solution is more {word} than the previous one.", "medium"),
            ("Given the circumstances, the outcome was {word}.", "advanced"),
            ("The {n} looks {word} in this light.", "basic"),
            ("Her {adj} approach was both {word} and effective.", "advanced")
        ]
        
        # Adverb templates
        self.sentence_templates['adv'] = [
            ("He works {word}.", "basic"),
            ("She spoke {word} about the {n}.", "medium"),
            ("The system performed {word} well under pressure.", "advanced"),
            ("You should do it {word}.", "basic"),
            ("The team collaborated {word} on the project.", "medium"),
            ("Despite the challenges, they progressed {word}.", "advanced"),
            ("Time passes {word}.", "basic"),
            ("The algorithm processes data {word}.", "medium")
        ]
        
        # Default templates for other POS
        self.sentence_templates['default'] = [
            ("This involves the {word}.", "basic"),
            ("We considered the {word} in our analysis.", "medium"),
            ("The concept of {word} is central to our understanding.", "advanced"),
            ("I think about {word} often.", "basic"),
            ("The discussion focused on {word}.", "medium"),
            ("From a theoretical perspective, {word} is significant.", "advanced")
        ]
    
    def load_csv(self):
        """Load word database from CSV file"""
        try:
            if not os.path.exists(self.csv_path):
                # Create sample database if doesn't exist
                self.create_sample_database()
                return True
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row['word'].lower().strip()
                    self.word_data[word] = {
                        'word': word,
                        'pos': row.get('pos', 'n').lower(),
                        'category': row.get('category', 'general'),
                        'frequency': float(row.get('frequency', 0.5)),
                        'sentences': json.loads(row.get('sentences', '[]')),
                        'synonyms': json.loads(row.get('synonyms', '[]')),
                        'antonyms': json.loads(row.get('antonyms', '[]')),
                        'definition': row.get('definition', ''),
                        'examples': json.loads(row.get('examples', '[]'))
                    }
                    
                    # Index by POS
                    pos = row.get('pos', 'n').lower()
                    self.pos_index[pos].append(word)
                    
                    # Index synonyms
                    synonyms = json.loads(row.get('synonyms', '[]'))
                    self.synonyms[word] = synonyms
            
            print(f"Loaded {len(self.word_data)} words from database")
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def create_sample_database(self):
        """Create sample CSV database if it doesn't exist"""
        sample_data = [
            # Word, POS, Category, Frequency, Sentences, Synonyms, Antonyms, Definition, Examples
            ["headphones", "n", "electronics", 0.7, 
             '["The headphones produce excellent sound quality.", "I wear headphones when working.", "Wireless headphones are convenient."]',
             '["earphones", "headset", "earbuds"]', '[]',
             "A pair of listening devices worn over the ears",
             '["She put on her headphones to listen to music.", "The headphones have noise-cancelling features."]'],
            
            ["computer", "n", "technology", 0.9,
             '["The computer processes data quickly.", "My computer has 16GB of RAM.", "Quantum computers represent the future."]',
             '["PC", "machine", "device"]', '[]',
             "An electronic device for storing and processing data",
             '["He turned on the computer to start working.", "The computer crashed during the update."]'],
            
            ["learn", "v", "education", 0.8,
             '["I learn new things every day.", "Children learn quickly through play.", "We must learn from our mistakes."]',
             '["study", "acquire", "grasp"]', '["forget", "ignore"]',
             "To gain knowledge or skill through study or experience",
             '["She wants to learn programming.", "The goal is to learn from experience."]'],
            
            ["beautiful", "adj", "description", 0.6,
             '["The sunset was beautiful.", "She has a beautiful voice.", "It was a beautiful moment."]',
             '["gorgeous", "stunning", "lovely"]', '["ugly", "plain"]',
             "Pleasing the senses or mind aesthetically",
             '["The garden looks beautiful in spring.", "What a beautiful painting!"]'],
            
            ["quickly", "adv", "manner", 0.5,
             '["He ran quickly to catch the bus.", "The project progressed quickly.", "Time passes quickly when you\'re having fun."]',
             '["rapidly", "swiftly", "speedily"]', '["slowly", "gradually"]',
             "At a fast speed; rapidly",
             '["She quickly finished her homework.", "The situation changed quickly."]']
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'pos', 'category', 'frequency', 'sentences', 
                           'synonyms', 'antonyms', 'definition', 'examples'])
            writer.writerows(sample_data)
        
        print(f"Created sample database with {len(sample_data)} words")
        self.load_csv()
    
    def search_word(self, word):
        """Search for a word in the database"""
        word_lower = word.lower().strip()
        
        # Direct match
        if word_lower in self.word_data:
            return self.word_data[word_lower]
        
        # Search in synonyms
        for main_word, synonyms in self.synonyms.items():
            if word_lower in [s.lower() for s in synonyms]:
                return self.word_data[main_word]
        
        # No match found
        return None
    
    def get_random_sentence(self, word_data, complexity="medium"):
        """Get a random sentence for a word with given complexity"""
        if not word_data:
            return None
        
        word = word_data['word']
        pos = word_data['pos']
        
        # Get templates for this POS
        templates = self.sentence_templates.get(pos, self.sentence_templates['default'])
        
        # Filter by complexity
        filtered_templates = [t for t in templates if t[1] == complexity]
        if not filtered_templates:
            filtered_templates = templates
        
        # Choose random template
        template, _ = random.choice(filtered_templates)
        
        # Fill template with appropriate words
        sentence = self.fill_template(template, word_data)
        
        return sentence
    
    def fill_template(self, template, word_data):
        """Fill a sentence template with appropriate words"""
        word = word_data['word']
        pos = word_data['pos']
        
        # Replace {word} with the actual word
        result = template.replace("{word}", word)
        
        # Get random adjective
        adjectives = self.pos_index.get('adj', [])
        if adjectives:
            adj = random.choice(adjectives)
            result = result.replace("{adj}", adj)
            result = result.replace("{adj2}", random.choice([a for a in adjectives if a != adj]))
        
        # Get random noun
        nouns = self.pos_index.get('n', [])
        if nouns:
            noun = random.choice(nouns)
            result = result.replace("{n}", noun)
            result = result.replace("{n2}", random.choice([n for n in nouns if n != noun]))
        
        return result
    
    def get_pos_name(self, pos_code):
        """Get full part of speech name from code"""
        return self.PARTS_OF_SPEECH.get(pos_code, "Unknown")

# ============================================================================
# BACKGROUND SEARCH THREAD
# ============================================================================

class WordSearchThread(QThread):
    """Thread for background word search in database"""
    
    # Signal to emit when search is complete
    search_complete = pyqtSignal(dict, dict, str)
    search_failed = pyqtSignal(str)
    
    def __init__(self, word, database):
        super().__init__()
        self.word = word
        self.database = database
    
    def run(self):
        """Run the search in background thread"""
        try:
            # Simulate search time (for demonstration)
            time.sleep(0.1)
            
            # Search for word in database
            word_data = self.database.search_word(self.word)
            
            if word_data:
                # Generate sentence with random complexity
                complexities = ["basic", "medium", "advanced"]
                complexity = random.choice(complexities)
                sentence = self.database.get_random_sentence(word_data, complexity)
                
                # Prepare result
                result = {
                    'word': self.word,
                    'word_data': word_data,
                    'sentence': sentence,
                    'complexity': complexity,
                    'found_in_db': True
                }
                
                self.search_complete.emit(result, word_data, sentence)
            else:
                result = {
                    'word': self.word,
                    'found_in_db': False
                }
                self.search_failed.emit(self.word)
                
        except Exception as e:
            print(f"Search error: {e}")
            self.search_failed.emit(self.word)

# ============================================================================
# ENHANCED NLP PROCESSOR
# ============================================================================

class EnhancedNLPProcessor(NLPProcessor):
    """Enhanced NLP processor with CSV database support"""
    
    def __init__(self, database=None):
        super().__init__()
        self.database = database
        self.search_thread = None
        
    def generate_sentence_from_db(self, word, nn_output):
        """Generate sentences using CSV database"""
        if not self.database:
            return self.generate_sentences(word, nn_output)
            
        # Search for word in database
        word_data = self.database.search_word(word)
        
        if word_data:
            # Use NN output to select complexity
            complexity_idx = np.argmax(nn_output[:3])
            complexities = ['basic', 'medium', 'advanced']
            complexity = complexities[complexity_idx]
            
            # Get sentence from database
            sentence = self.database.get_random_sentence(word_data, complexity)
            
            # Get word type from database
            pos_code = word_data['pos']
            word_type = self.database.get_pos_name(pos_code)
            
            return {
                'primary_sentence': sentence,
                'word_type': word_type,
                'complexity': complexity,
                'confidence': float(np.max(nn_output)),
                'found_in_db': True,
                'word_data': word_data
            }
        else:
            # Fall back to original method
            result = self.generate_sentences(word, nn_output)
            result['found_in_db'] = False
            return result
    
    def start_background_search(self, word, nn_output, callback):
        """Start background search for word"""
        if not self.database:
            # If no database, use immediate fallback
            result = self.generate_sentences(word, nn_output)
            result['found_in_db'] = False
            callback(result, None, result['primary_sentence'])
            return
            
        self.search_thread = WordSearchThread(word, self.database)
        self.search_thread.search_complete.connect(callback)
        self.search_thread.search_failed.connect(
            lambda: self.handle_search_failed(word, nn_output, callback)
        )
        self.search_thread.start()
    
    def handle_search_failed(self, word, nn_output, callback):
        """Handle case when word is not found in database"""
        # Fall back to original method
        result = self.generate_sentences(word, nn_output)
        result['found_in_db'] = False
        callback(result, None, result['primary_sentence'])

# ============================================================================
# DEBUG ENGINE & PERFORMANCE TRACKER
# ============================================================================

@dataclass
class DebugRecord:
    """Structure for debug information"""
    timestamp: str
    input_word: str
    token_vector: List[float]
    layer_activations: List[Dict]
    weight_stats: Dict
    processing_time: float
    fps: float
    warnings: List[str]
    errors: List[str]
    
class DebugEngine:
    """Handles debug logging and performance tracking"""
    
    def __init__(self, max_records=100):
        self.records = deque(maxlen=max_records)
        self.fps_history = deque(maxlen=60)
        self.start_time = time.time()
        self.frame_count = 0
        
    def log_processing(self, input_word, token_vector, layer_activations, 
                      weight_stats, processing_time):
        """Log a processing event"""
        current_fps = self.calculate_fps()
        
        record = DebugRecord(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            input_word=input_word,
            token_vector=token_vector.tolist() if hasattr(token_vector, 'tolist') else token_vector,
            layer_activations=layer_activations,
            weight_stats=weight_stats,
            processing_time=processing_time,
            fps=current_fps,
            warnings=self.generate_warnings(weight_stats, processing_time),
            errors=[]
        )
        
        self.records.append(record)
        return record
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        self.fps_history.append(current_fps)
        return current_fps
    
    def generate_warnings(self, weight_stats, processing_time):
        """Generate warnings based on stats"""
        warnings = []
        
        if weight_stats['zero_weights'] > weight_stats['total_weights'] * 0.1:
            warnings.append(f"High zero weights: {weight_stats['zero_weights']}/{weight_stats['total_weights']}")
        
        if processing_time > 100:  # ms
            warnings.append(f"Slow processing: {processing_time:.1f}ms")
        
        if weight_stats['std_weight'] < 0.01:
            warnings.append("Low weight variance - network may be underfitting")
            
        return warnings
    
    def get_performance_summary(self):
        """Get performance summary"""
        if not self.fps_history:
            return {"avg_fps": 0, "min_fps": 0, "max_fps": 0}
        
        return {
            "avg_fps": np.mean(self.fps_history),
            "min_fps": np.min(self.fps_history),
            "max_fps": np.max(self.fps_history),
            "total_processes": len(self.records)
        }
    
    def export_report(self, format='txt'):
        """Export debug report"""
        if format == 'json':
            report = {
                'timestamp': datetime.now().isoformat(),
                'records': [record.__dict__ for record in self.records],
                'performance': self.get_performance_summary()
            }
            return json.dumps(report, indent=2)
        else:  # txt
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("NEURAL NETWORK DEBUG REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now()}")
            report_lines.append(f"Total Records: {len(self.records)}")
            report_lines.append("\nPERFORMANCE SUMMARY:")
            perf = self.get_performance_summary()
            report_lines.append(f"  Average FPS: {perf['avg_fps']:.1f}")
            report_lines.append(f"  Min FPS: {perf['min_fps']:.1f}")
            report_lines.append(f"  Max FPS: {perf['max_fps']:.1f}")
            
            if self.records:
                report_lines.append("\nLATEST PROCESSING:")
                latest = self.records[-1]
                report_lines.append(f"  Word: {latest.input_word}")
                report_lines.append(f"  Time: {latest.processing_time:.2f} ms")
                report_lines.append(f"  FPS: {latest.fps:.1f}")
                report_lines.append(f"  Weights Used: {latest.weight_stats['total_weights']}")
                if latest.warnings:
                    report_lines.append(f"  Warnings: {', '.join(latest.warnings)}")
            
            return "\n".join(report_lines)

# ============================================================================
# VISUALIZATION WIDGETS (Same as before)
# ============================================================================

class NeuralCanvas(QWidget):
    """Widget for visualizing neural network with real weights"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.network = None
        self.animation_speed = 1.0
        self.is_animating = False
        self.current_layer = 0
        self.activation_levels = []
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_step)
        self.dark_mode = True
        
        # Animation state
        self.animation_progress = 0.0
        self.activation_pulse = 0.0
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.current_fps = 0
        
        self.setMinimumSize(800, 500)
        
    def set_network(self, network):
        """Set the neural network to visualize"""
        self.network = network
        self.activation_levels = []
        self.update()
        
    def start_animation(self):
        """Start the animation"""
        self.is_animating = True
        self.current_layer = 0
        self.animation_progress = 0.0
        self.activation_pulse = 1.0
        self.animation_timer.start(int(16 / self.animation_speed))
        
    def stop_animation(self):
        """Stop the animation"""
        self.is_animating = False
        self.animation_timer.stop()
        
    def animate_step(self):
        """Animation step"""
        self.animation_progress += 0.02 * self.animation_speed
        
        if self.animation_progress >= 1.0:
            self.current_layer += 1
            self.animation_progress = 0.0
            self.activation_pulse = 1.0
            
            if self.current_layer >= len(self.network.layer_sizes):
                self.stop_animation()
                self.current_layer = len(self.network.layer_sizes) - 1
        
        # Pulse effect
        self.activation_pulse *= 0.95
        
        # Calculate FPS
        current_time = time.time()
        self.current_fps = 1.0 / (current_time - self.last_frame_time) if current_time > self.last_frame_time else 0
        self.last_frame_time = current_time
        
        self.update()
        
    def set_animation_speed(self, speed):
        """Set animation speed multiplier"""
        self.animation_speed = speed
        if self.is_animating:
            self.animation_timer.setInterval(int(16 / speed))
            
    def toggle_dark_mode(self, dark):
        """Toggle dark/light mode"""
        self.dark_mode = dark
        self.update()
        
    def paintEvent(self, event):
        """Paint the neural network visualization"""
        if not self.network:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background
        if self.dark_mode:
            painter.fillRect(self.rect(), QColor(20, 25, 35))
            text_color = QColor(220, 220, 220)
            neuron_color = QColor(100, 200, 255)
        else:
            painter.fillRect(self.rect(), QColor(240, 245, 255))
            text_color = QColor(50, 50, 50)
            neuron_color = QColor(0, 100, 200)
            
        # Calculate layout
        layer_count = len(self.network.layer_sizes)
        neuron_radius = 20
        layer_spacing = self.width() / (layer_count + 1)
        
        # Draw connections with weights
        for layer_idx in range(layer_count - 1):
            neurons_current = self.network.layer_sizes[layer_idx]
            neurons_next = self.network.layer_sizes[layer_idx + 1]
            
            x_current = (layer_idx + 1) * layer_spacing
            x_next = (layer_idx + 2) * layer_spacing
            
            for i in range(neurons_current):
                y_current = self.height() / (neurons_current + 1) * (i + 1)
                
                for j in range(neurons_next):
                    y_next = self.height() / (neurons_next + 1) * (j + 1)
                    
                    # Get weight value
                    weight = self.network.weights[layer_idx][i, j]
                    
                    # Determine line style based on weight
                    if weight > 0:
                        if self.dark_mode:
                            line_color = QColor(0, 200, 255, min(255, int(abs(weight) * 200 + 55)))
                        else:
                            line_color = QColor(0, 150, 255, min(255, int(abs(weight) * 150 + 55)))
                    else:
                        if self.dark_mode:
                            line_color = QColor(255, 100, 100, min(255, int(abs(weight) * 200 + 55)))
                        else:
                            line_color = QColor(255, 50, 50, min(255, int(abs(weight) * 150 + 55)))
                    
                    # Animation effects
                    if self.is_animating and layer_idx == self.current_layer:
                        progress = self.animation_progress
                        if progress > (j / neurons_next) and progress < ((j + 1) / neurons_next):
                            line_width = 3 + 2 * self.activation_pulse
                            line_color.setAlpha(min(255, line_color.alpha() + int(100 * self.activation_pulse)))
                        else:
                            line_width = 1
                    else:
                        line_width = max(0.5, min(3, abs(weight) * 2))
                    
                    # Draw connection
                    pen = QPen(line_color, line_width)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawLine(int(x_current), int(y_current), int(x_next), int(y_next))
        
        # Draw neurons
        for layer_idx in range(layer_count):
            neurons = self.network.layer_sizes[layer_idx]
            x = (layer_idx + 1) * layer_spacing
            
            for i in range(neurons):
                y = self.height() / (neurons + 1) * (i + 1)
                
                # Determine activation level
                if layer_idx < len(self.network.activations):
                    if layer_idx == 0:  # Input layer
                        activation = self.network.activations[layer_idx][i] if i < len(self.network.activations[layer_idx]) else 0
                    else:
                        activation = self.network.activations[layer_idx][i] if i < len(self.network.activations[layer_idx]) else 0
                else:
                    activation = 0
                
                # Draw neuron
                neuron_glow = activation * 0.5 + 0.5
                if self.dark_mode:
                    base_color = neuron_color
                else:
                    base_color = QColor(0, 100, 200)
                
                # Animation pulse
                if self.is_animating and layer_idx == self.current_layer:
                    pulse = self.activation_pulse
                else:
                    pulse = 0
                
                glow_color = QColor(
                    min(255, base_color.red() + int(55 * neuron_glow) + int(100 * pulse)),
                    min(255, base_color.green() + int(55 * neuron_glow) + int(100 * pulse)),
                    min(255, base_color.blue() + int(55 * neuron_glow) + int(100 * pulse))
                )
                
                painter.setBrush(QBrush(glow_color))
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
                painter.drawEllipse(int(x - neuron_radius), int(y - neuron_radius), 
                                   neuron_radius * 2, neuron_radius * 2)
                
                # Draw activation value
                painter.setPen(text_color)
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(int(x - 15), int(y + 5), f"{activation:.2f}")
        
        # Draw layer labels
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        layer_names = ["Input"] + [f"Hidden {i+1}" for i in range(len(self.network.hidden_layers))] + ["Output"]
        for i, name in enumerate(layer_names):
            x = (i + 1) * layer_spacing
            painter.drawText(int(x - 30), 20, name)
            
        # Draw FPS
        painter.drawText(10, 20, f"FPS: {self.current_fps:.1f}")
        
        # Draw animation progress
        if self.is_animating:
            painter.setPen(QColor(100, 255, 100))
            painter.drawText(10, 40, f"Animating Layer: {self.current_layer + 1}")
            
        painter.end()

# ============================================================================
# CHAT PANEL
# ============================================================================

class ChatPanel(QTextEdit):
    """Panel for displaying explainability text"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 35, 45, 200);
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                color: #ddd;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
    def add_message(self, sender, message):
        """Add a message to the chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append(f"[{timestamp}] <b>{sender}:</b> {message}")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

# ============================================================================
# ENHANCED MAIN WINDOW
# ============================================================================

class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with database support"""
    
    def __init__(self):
        super().__init__()
        
        # Load database first
        self.word_database = WordDatabase()
        self.word_database.load_csv()
        
        # Initialize neural network and enhanced NLP processor
        self.neural_network = NeuralNetwork()
        self.nlp_processor = EnhancedNLPProcessor(self.word_database)
        self.debug_engine = DebugEngine()
        
        self.init_ui()
        self.last_process_time = 0
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🧠 Interactive Neural Network NLP Simulator v2.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ===== TOP CONTROL BAR =====
        control_bar = QHBoxLayout()
        
        # Process controls
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("Enter a word (e.g., 'headphones', 'learn', 'beautiful')...")
        self.word_input.setMinimumWidth(300)
        self.word_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #555;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        
        process_btn = QPushButton("🚀 Process Word")
        process_btn.clicked.connect(self.process_word)
        process_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Database status
        self.db_status = QLabel(f"📚 {len(self.word_database.word_data)} words")
        self.db_status.setToolTip("Words in database")
        self.db_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # Animation controls
        self.animation_slider = QSlider(Qt.Horizontal)
        self.animation_slider.setRange(25, 400)  # 0.25x to 4.0x
        self.animation_slider.setValue(100)  # 1.0x
        self.animation_slider.valueChanged.connect(self.update_animation_speed)
        
        speed_label = QLabel("Speed:")
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # Dark/light mode toggle
        self.dark_mode_btn = QPushButton("🌙 Dark Mode")
        self.dark_mode_btn.setCheckable(True)
        self.dark_mode_btn.setChecked(True)
        self.dark_mode_btn.clicked.connect(self.toggle_dark_mode)
        
        control_bar.addWidget(QLabel("Input:"))
        control_bar.addWidget(self.word_input)
        control_bar.addWidget(process_btn)
        control_bar.addWidget(self.db_status)
        control_bar.addWidget(speed_label)
        control_bar.addWidget(self.animation_slider)
        control_bar.addWidget(self.fps_label)
        control_bar.addWidget(self.dark_mode_btn)
        control_bar.addStretch()
        
        main_layout.addLayout(control_bar)
        
        # ===== NEURAL NETWORK VISUALIZATION =====
        self.neural_canvas = NeuralCanvas()
        main_layout.addWidget(self.neural_canvas, 3)
        
        # ===== CHAT EXPLAINABILITY PANELS =====
        chat_layout = QHBoxLayout()
        
        self.performance_chat = ChatPanel("Performance")
        self.structure_chat = ChatPanel("Structural Flow")
        self.logic_chat = ChatPanel("Internal Logic")
        
        chat_layout.addWidget(self.performance_chat)
        chat_layout.addWidget(self.structure_chat)
        chat_layout.addWidget(self.logic_chat)
        
        main_layout.addLayout(chat_layout)
        
        # ===== OUTPUT DISPLAY =====
        output_layout = QHBoxLayout()
        
        # Sentence output
        self.sentence_output = QTextEdit()
        self.sentence_output.setReadOnly(True)
        self.sentence_output.setMaximumHeight(120)
        self.sentence_output.setStyleSheet("""
            QTextEdit {
                background-color: rgba(40, 45, 60, 200);
                border: 1px solid #666;
                border-radius: 5px;
                padding: 10px;
                color: #fff;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # Statistics display
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(120)
        self.stats_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 40, 50, 200);
                border: 1px solid #666;
                border-radius: 5px;
                padding: 10px;
                color: #aaddff;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        
        output_layout.addWidget(self.sentence_output, 2)
        output_layout.addWidget(self.stats_display, 1)
        
        main_layout.addLayout(output_layout)
        
        # ===== DEBUG PANEL =====
        debug_group = QGroupBox("Debug Panel")
        debug_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ff5555;
            }
        """)
        
        debug_layout = QVBoxLayout()
        
        # Debug text area
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumHeight(150)
        self.debug_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #444;
            }
        """)
        
        # Debug controls
        debug_controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.debug_text.clear)
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_debug_report)
        
        debug_controls.addWidget(clear_btn)
        debug_controls.addWidget(export_btn)
        debug_controls.addStretch()
        
        debug_layout.addWidget(self.debug_text)
        debug_layout.addLayout(debug_controls)
        debug_group.setLayout(debug_layout)
        
        main_layout.addWidget(debug_group)
        
        # Status bar
        self.statusBar().showMessage("Ready. Enter a word and click 'Process Word' to begin.")
        
        # Initialize neural canvas
        self.neural_canvas.set_network(self.neural_network)
        
        # Add database menu
        self.add_database_menu()
        
    def add_database_menu(self):
        """Add database management menu"""
        menubar = self.menuBar()
        
        # Database menu
        database_menu = menubar.addMenu("Database")
        
        manage_action = QAction("📊 Manage Database", self)
        manage_action.triggered.connect(self.open_database_dialog)
        database_menu.addAction(manage_action)
        
        refresh_action = QAction("🔄 Refresh Database", self)
        refresh_action.triggered.connect(self.refresh_database)
        database_menu.addAction(refresh_action)
        
        database_menu.addSeparator()
        
        stats_action = QAction("📈 Show Statistics", self)
        stats_action.triggered.connect(self.show_database_stats)
        database_menu.addAction(stats_action)
        
        add_word_action = QAction("➕ Add Sample Words", self)
        add_word_action.triggered.connect(self.add_sample_words)
        database_menu.addAction(add_word_action)
    
    def process_word(self):
        """Enhanced process word with database support"""
        word = self.word_input.text().strip()
        if not word:
            QMessageBox.warning(self, "Input Error", "Please enter a word to process.")
            return
        
        # Update UI
        self.statusBar().showMessage(f"Searching for '{word}' in database...")
        
        # Process word through NLP to get token vector
        token_vector = self.nlp_processor.process_word(word)
        
        # Neural network forward propagation
        output, layer_activations = self.neural_network.forward_propagate(token_vector)
        
        # Get weight statistics
        weight_stats = self.neural_network.get_weight_statistics()
        
        # Log to debug engine
        debug_record = self.debug_engine.log_processing(
            word, token_vector, layer_activations, 
            weight_stats, self.neural_network.processing_time
        )
        
        # Start background search
        self.statusBar().showMessage(f"Background search for '{word}'...")
        self.nlp_processor.start_background_search(
            word, output, 
            lambda result, word_data, sentence: self.on_search_complete(
                word, result, word_data, sentence, token_vector, 
                layer_activations, weight_stats, debug_record, output
            )
        )
        
        # Start animation immediately (don't wait for search)
        self.neural_canvas.start_animation()
    
    def on_search_complete(self, word, result, word_data, sentence, token_vector, 
                          layer_activations, weight_stats, debug_record, nn_output):
        """Handle completion of background search"""
        
        # Update sentence display
        if result['found_in_db']:
            # Use database-generated sentence
            sentence_data = result
            if word_data:
                sentence_data['word_type'] = self.word_database.get_pos_name(word_data['pos'])
        else:
            # Use original method
            sentence_data = self.nlp_processor.generate_sentences(word, nn_output)
            sentence_data['found_in_db'] = False
        
        # Update UI components
        self.update_sentence_display(word, sentence_data)
        self.update_stats_display(debug_record, weight_stats)
        self.update_chat_panels(word, layer_activations, sentence_data)
        self.update_debug_panel(debug_record)
        
        # Add database info to chat
        if result['found_in_db']:
            self.structure_chat.add_message("Database", 
                f"Word found in database (POS: {sentence_data.get('word_type', 'Unknown')})")
            self.logic_chat.add_message("Database",
                f"Using pre-defined sentence template for {sentence_data.get('word_type', 'Unknown')}")
        else:
            self.structure_chat.add_message("Database", 
                f"Word not found in database - using generative templates")
            self.logic_chat.add_message("Database",
                "Using fallback generative sentence creation")
        
        # Update status
        source = "database" if result['found_in_db'] else "generative"
        self.statusBar().showMessage(
            f"Processed '{word}' from {source} in {self.neural_network.processing_time:.1f}ms"
        )
        
        # Update FPS label
        self.fps_label.setText(f"FPS: {debug_record.fps:.1f}")
    
    def update_sentence_display(self, word, sentence_data):
        """Enhanced sentence display with database info"""
        primary = sentence_data['primary_sentence']
        complexity = sentence_data.get('complexity', 'medium').upper()
        word_type = sentence_data.get('word_type', 'Unknown')
        source = "📚 Database" if sentence_data.get('found_in_db', False) else "🤖 Generated"
        
        html = f"""
        <div style='text-align: center;'>
            <h3 style='color: #4CAF50;'>Generated Sentence ({source}):</h3>
            <p style='font-size: 16px; color: #fff;'><b>"{primary}"</b></p>
            <p style='color: #aaa;'>
                Word: <b style='color: #FFD700;'>{word}</b> | 
                Type: <b style='color: #FFD700;'>{word_type}</b> | 
                Complexity: <b style='color: #FFD700;'>{complexity}</b>
            </p>
        </div>
        """
        
        self.sentence_output.setHtml(html)
    
    def update_stats_display(self, debug_record, weight_stats):
        """Update the statistics display"""
        stats_text = f"""
╔═══════════════════════════════════════╗
║        NEURAL NETWORK STATS           ║
╠═══════════════════════════════════════╣
║ Processing Time: {debug_record.processing_time:6.1f} ms
║ FPS:             {debug_record.fps:6.1f}
║ Total Weights:   {weight_stats['total_weights']:6d}
║ ──────────────────────────────────────
║ Mean Weight:     {weight_stats['mean_weight']:6.3f}
║ Weight Std:      {weight_stats['std_weight']:6.3f}
║ Max Weight:      {weight_stats['max_weight']:6.3f}
║ Min Weight:      {weight_stats['min_weight']:6.3f}
║ ──────────────────────────────────────
║ Positive:        {weight_stats['positive_weights']:6d}
║ Negative:        {weight_stats['negative_weights']:6d}
║ Zero:            {weight_stats['zero_weights']:6d}
╚═══════════════════════════════════════╝
        """
        
        self.stats_display.setText(stats_text)
        
    def update_chat_panels(self, word, layer_activations, sentence_data):
        """Update the explainability chat panels"""
        # Performance Chat
        self.performance_chat.clear()
        self.performance_chat.add_message("System", f"Processing completed for '{word}'")
        self.performance_chat.add_message("Performance", 
            f"Frame rate: {self.debug_engine.calculate_fps():.1f} FPS")
        
        if self.debug_engine.fps_history:
            avg_fps = np.mean(self.debug_engine.fps_history)
            self.performance_chat.add_message("Performance",
                f"Average FPS: {avg_fps:.1f} (Min: {np.min(self.debug_engine.fps_history):.1f}, "
                f"Max: {np.max(self.debug_engine.fps_history):.1f})")
        
        # Structural Flow Chat
        self.structure_chat.clear()
        self.structure_chat.add_message("System", "Neural Network Architecture:")
        self.structure_chat.add_message("Structure", 
            f"Layers: Input({self.neural_network.input_size}) → "
            f"Hidden{self.neural_network.hidden_layers} → "
            f"Output({self.neural_network.output_size})")
        
        for i, activation in enumerate(layer_activations):
            self.structure_chat.add_message(f"Layer {i+1}",
                f"Max activation: {activation['max_activation']:.3f}, "
                f"Avg: {activation['avg_activation']:.3f}")
        
        # Internal Logic Chat
        self.logic_chat.clear()
        self.logic_chat.add_message("System", f"Word Analysis: '{word}'")
        self.logic_chat.add_message("NLP Engine", 
            f"Word type detected: {sentence_data.get('word_type', 'Unknown')}")
        self.logic_chat.add_message("Logic", 
            f"Selected complexity: {sentence_data.get('complexity', 'medium')} "
            f"(confidence: {sentence_data.get('confidence', 0):.2%})")
        self.logic_chat.add_message("Logic", 
            f"Source: {'Database' if sentence_data.get('found_in_db') else 'Generative'}")
        
    def update_debug_panel(self, debug_record):
        """Update the debug panel with latest information"""
        timestamp = debug_record.timestamp
        
        debug_text = f"""
[{timestamp}] PROCESS_START: Word='{debug_record.input_word}'
[{timestamp}] TOKENIZATION: Vector length={len(debug_record.token_vector)}
[{timestamp}] NEURAL_PROPAGATION: Layers={len(debug_record.layer_activations)+1}
[{timestamp}] TIMING: Processing={debug_record.processing_time:.1f}ms, FPS={debug_record.fps:.1f}
[{timestamp}] WEIGHTS: Total={debug_record.weight_stats['total_weights']}, "
    "Pos={debug_record.weight_stats['positive_weights']}, Neg={debug_record.weight_stats['negative_weights']}
"""
        
        if debug_record.warnings:
            for warning in debug_record.warnings:
                debug_text += f"[{timestamp}] WARNING: {warning}\n"
        
        debug_text += f"[{timestamp}] PROCESS_COMPLETE: Status=OK\n"
        
        self.debug_text.append(debug_text)
        
    def update_animation_speed(self):
        """Update animation speed based on slider"""
        speed = self.animation_slider.value() / 100.0
        self.neural_canvas.set_animation_speed(speed)
        
    def toggle_dark_mode(self, checked):
        """Toggle between dark and light mode"""
        if checked:
            self.dark_mode_btn.setText("🌙 Dark Mode")
            self.neural_canvas.toggle_dark_mode(True)
        else:
            self.dark_mode_btn.setText("☀️ Light Mode")
            self.neural_canvas.toggle_dark_mode(False)
            
    def export_debug_report(self):
        """Export debug report to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Debug Report", "", "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            if file_path.endswith('.json'):
                report = self.debug_engine.export_report('json')
            else:
                report = self.debug_engine.export_report('txt')
                
            with open(file_path, 'w') as f:
                f.write(report)
                
            QMessageBox.information(self, "Export Complete", 
                                  f"Debug report exported to:\n{file_path}")
    
    def open_database_dialog(self):
        """Open database management dialog"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Database Information")
        dialog.setGeometry(200, 200, 400, 300)
        
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # Get database stats
        db = self.word_database
        stats = f"""
Database Statistics:
====================
Total Words: {len(db.word_data)}

Words by Part of Speech:
"""
        pos_counts = {}
        for word, data in db.word_data.items():
            pos = data['pos']
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        for pos, count in pos_counts.items():
            stats += f"  {db.get_pos_name(pos)} ({pos}): {count} words\n"
        
        # Get most common categories
        categories = {}
        for word, data in db.word_data.items():
            cat = data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        stats += "\nTop Categories:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            stats += f"  {cat}: {count} words\n"
        
        info_text.setText(stats)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        
        layout.addWidget(info_text)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def refresh_database(self):
        """Refresh database from file"""
        self.word_database.load_csv()
        self.db_status.setText(f"📚 {len(self.word_database.word_data)} words")
        QMessageBox.information(self, "Database Refreshed", 
                              f"Loaded {len(self.word_database.word_data)} words.")
    
    def show_database_stats(self):
        """Show database statistics"""
        self.open_database_dialog()
    
    def add_sample_words(self):
        """Add more sample words to database"""
        sample_words = [
            ["innovation", "n", "business", 0.6,
             '["Technological innovation drives economic growth.", "The company\'s innovation revolutionized the market.", "Continuous innovation keeps us competitive."]',
             '["invention", "novelty", "breakthrough"]', '[]',
             "The introduction of new ideas, methods, or products",
             '["Innovation requires creativity and risk-taking.", "The innovation department works on future projects."]'],
            
            ["collaborate", "v", "work", 0.7,
             '["The two companies collaborate on research.", "We need to collaborate to find a solution.", "Teams collaborate using online tools."]',
             '["cooperate", "work together", "team up"]', '["compete", "work alone"]',
             "Work jointly on an activity or project",
             '["Collaborating with experts improves outcomes.", "We should collaborate rather than compete."]'],
            
            ["efficiently", "adv", "manner", 0.5,
             '["The team worked efficiently to complete the project.", "The machine processes data efficiently.", "She manages her time efficiently."]',
             '["effectively", "productively", "competently"]', '["inefficiently", "poorly"]',
             "In a way that achieves maximum productivity",
             '["The new software runs more efficiently.", "We need to use resources efficiently."]']
        ]
        
        # Add to CSV file
        file_exists = os.path.exists(self.word_database.csv_path)
        mode = 'a' if file_exists else 'w'
        
        with open(self.word_database.csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['word', 'pos', 'category', 'frequency', 'sentences', 
                               'synonyms', 'antonyms', 'definition', 'examples'])
            writer.writerows(sample_words)
        
        # Reload database
        self.refresh_database()
        QMessageBox.information(self, "Sample Words Added", 
                              f"Added {len(sample_words)} sample words to database.")

# ============================================================================
# DATABASE MANAGEMENT DIALOG
# ============================================================================

class DatabaseDialog(QDialog):
    """Dialog for managing word database"""
    
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("Word Database Manager")
        self.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        # Database info
        info_group = QGroupBox("Database Information")
        info_layout = QFormLayout()
        
        self.total_words = QLabel(str(len(database.word_data)))
        self.words_by_pos = QTextEdit()
        self.words_by_pos.setReadOnly(True)
        
        # Count words by POS
        pos_counts = {}
        for word, data in database.word_data.items():
            pos = data['pos']
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        pos_text = "\n".join([f"{database.get_pos_name(pos)} ({pos}): {count}"
                            for pos, count in pos_counts.items()])
        self.words_by_pos.setText(pos_text)
        
        info_layout.addRow("Total Words:", self.total_words)
        info_layout.addRow("Words by POS:", self.words_by_pos)
        info_group.setLayout(info_layout)
        
        # Search functionality
        search_group = QGroupBox("Search Word")
        search_layout = QVBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter word to search...")
        self.search_result = QTextEdit()
        self.search_result.setReadOnly(True)
        self.search_result.setMaximumHeight(150)
        
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_word)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(self.search_result)
        search_group.setLayout(search_layout)
        
        # Add word functionality
        add_group = QGroupBox("Add New Word")
        add_layout = QFormLayout()
        
        self.new_word = QLineEdit()
        self.new_pos = QComboBox()
        self.new_pos.addItems(["n", "v", "adj", "adv", "prep", "conj", "pron", "det"])
        self.new_category = QLineEdit("general")
        self.new_sentence = QTextEdit()
        self.new_sentence.setMaximumHeight(80)
        
        add_layout.addRow("Word:", self.new_word)
        add_layout.addRow("Part of Speech:", self.new_pos)
        add_layout.addRow("Category:", self.new_category)
        add_layout.addRow("Example Sentence:", self.new_sentence)
        
        add_btn = QPushButton("Add to Database")
        add_btn.clicked.connect(self.add_word)
        
        add_layout.addRow(add_btn)
        add_group.setLayout(add_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        export_btn = QPushButton("Export Database")
        export_btn.clicked.connect(self.export_database)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(export_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        # Add all to main layout
        layout.addWidget(info_group)
        layout.addWidget(search_group)
        layout.addWidget(add_group)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def search_word(self):
        """Search for a word in database"""
        word = self.search_input.text().strip()
        if not word:
            return
        
        word_data = self.database.search_word(word)
        if word_data:
            result = f"""
Word: {word_data['word']}
Part of Speech: {self.database.get_pos_name(word_data['pos'])} ({word_data['pos']})
Category: {word_data['category']}
Definition: {word_data['definition']}

Synonyms: {', '.join(word_data['synonyms'])}
Examples: {'; '.join(word_data['examples'][:3])}
            """
        else:
            result = f"Word '{word}' not found in database."
        
        self.search_result.setText(result)
    
    def add_word(self):
        """Add a new word to database"""
        word = self.new_word.text().strip()
        pos = self.new_pos.currentText()
        category = self.new_category.text().strip()
        sentence = self.new_sentence.toPlainText().strip()
        
        if not word:
            QMessageBox.warning(self, "Input Error", "Please enter a word.")
            return
        
        # Add to database
        self.database.word_data[word.lower()] = {
            'word': word.lower(),
            'pos': pos,
            'category': category,
            'frequency': 0.5,
            'sentences': [sentence] if sentence else [],
            'synonyms': [],
            'antonyms': [],
            'definition': '',
            'examples': [sentence] if sentence else []
        }
        
        # Update index
        self.database.pos_index[pos].append(word.lower())
        
        # Update display
        self.total_words.setText(str(len(self.database.word_data)))
        
        QMessageBox.information(self, "Success", f"Word '{word}' added to database.")
    
    def export_database(self):
        """Export database to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Database", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['word', 'pos', 'category', 'frequency', 
                                   'sentences', 'synonyms', 'antonyms', 
                                   'definition', 'examples'])
                    
                    for word, data in self.database.word_data.items():
                        writer.writerow([
                            data['word'],
                            data['pos'],
                            data['category'],
                            data['frequency'],
                            json.dumps(data['sentences']),
                            json.dumps(data['synonyms']),
                            json.dumps(data['antonyms']),
                            data['definition'],
                            json.dumps(data['examples'])
                        ])
                
                QMessageBox.information(self, "Export Complete", 
                                      f"Database exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set application-wide dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 35, 45))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 30, 40))
    dark_palette.setColor(QPalette.AlternateBase, QColor(35, 40, 50))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(40, 45, 60))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(100, 200, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(100, 200, 255))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    # Check if CSV database exists, create if not
    if not os.path.exists("word_database.csv"):
        print("Creating sample word database...")
        db = WordDatabase()
        db.create_sample_database()
    
    # Create and show main window
    window = EnhancedMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
