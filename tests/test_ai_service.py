"""
Unit and Integration tests for AIService and RAGService.
Verifies:
1. BM25 RAG retrieval, scoring, grounding context, citations.
2. User health profile and conversation history context injection.
3. Offline fallback behavior when GEMINI_API_KEY is unset or empty.
4. Live or mocked Gemini 3.8 Flash API generation with google-genai SDK.
5. Graceful exception handling without crashing Flask or AIService.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from app import create_app
from models import db, User, UserProfile, NutritionTarget, ChatHistory
from services.rag_service import RAGService
from services.ai_service import AIService


@pytest.fixture
def test_app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        # Seed test user
        user = User(
            email='patient@example.com',
            full_name='Aarav Sharma'
        )
        user.set_password('SecurePass123!')
        db.session.add(user)
        db.session.flush()

        profile = UserProfile(
            user_id=user.id,
            age=32,
            gender='male',
            height_cm=175.0,
            weight_kg=72.0,
            dietary_preference='vegetarian',
            activity_level='moderate',
            food_allergies='peanuts',
            existing_conditions='pre-hypertension'
        )
        target = NutritionTarget(
            user_id=user.id,
            target_calories=2100,
            target_protein_g=75.0,
            target_carbs_g=280.0,
            target_fats_g=55.0
        )
        db.session.add(profile)
        db.session.add(target)
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()


def test_rag_service_retrieval():
    """Verify BM25 sparse retrieval loads documents and retrieves relevant chunks."""
    RAGService.reset()
    results = RAGService.retrieve('protein requirements vegetarian', top_k=3)
    assert len(results) > 0
    assert 'heading' in results[0]
    assert 'text' in results[0]
    assert 'source' in results[0]
    assert results[0]['score'] > 0


def test_rag_service_grounding_context_and_citations():
    """Verify grounding context builds formatted text and structured citations."""
    RAGService.reset()
    context, citations = RAGService.build_grounding_context('dietary fiber hypertension', top_k=3)
    assert len(context) > 0
    assert len(citations) > 0
    assert citations[0]['index'] == 1
    assert 'source' in citations[0]
    assert 'heading' in citations[0]
    assert '[Source 1:' in context


def test_user_health_context_injection(test_app):
    """Verify user context builder properly formats profile and nutritional targets."""
    with test_app.app_context():
        user = User.query.filter_by(email='patient@example.com').first()
        context = AIService.get_user_health_context(user.id)
        assert 'USER HEALTH PROFILE: Aarav Sharma' in context
        assert 'Age: 32 years' in context
        assert 'Dietary Preference: vegetarian' in context
        assert 'Daily Calorie Target: 2100' in context
        assert 'Food Allergies: peanuts' in context


def test_offline_fallback_when_api_key_missing(test_app):
    """Verify AIService gracefully falls back to RAG context when GEMINI_API_KEY is not set."""
    with test_app.app_context():
        user = User.query.filter_by(email='patient@example.com').first()

        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}, clear=False):
            result = AIService.generate_response(
                user_id=user.id,
                user_message='What are healthy high protein vegetarian foods in India?'
            )

            assert 'reply' in result
            assert 'citations' in result
            assert len(result['citations']) > 0
            # Offline response contains disclaimer and local knowledge marker
            assert 'clinical knowledge base' in result['reply'].lower()

            # Ensure both user message and assistant reply are persisted
            history = ChatHistory.query.filter_by(user_id=user.id).all()
            assert len(history) == 2
            assert history[0].sender == 'user'
            assert history[1].sender == 'assistant'


def test_api_failure_graceful_degradation(test_app):
    """Verify that when the Gemini API raises an exception, the application does not crash."""
    with test_app.app_context():
        user = User.query.filter_by(email='patient@example.com').first()

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'simulated_invalid_key'}, clear=False):
            with patch('google.genai.Client') as mock_client:
                mock_client.side_effect = Exception("API connection timed out or quota exceeded")

                result = AIService.generate_response(
                    user_id=user.id,
                    user_message='What should I eat for breakfast?'
                )

                assert 'reply' in result
                assert 'Live AI service encountered an issue' in result['reply']
                assert 'Response generated from local clinical knowledge base' in result['reply']


def test_gemini_38_flash_call(test_app):
    """Verify AIService calls gemini-3.8-flash with the google-genai client."""
    with test_app.app_context():
        user = User.query.filter_by(email='patient@example.com').first()

        mock_response = MagicMock()
        mock_response.text = "Here is evidence-based guidance for your vegetarian diet with approx 75g protein."

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key_12345'}, clear=False):
            with patch('google.genai.Client', return_value=mock_client_instance) as mock_client_cls:
                result = AIService.generate_response(
                    user_id=user.id,
                    user_message='How can I reach my 75g protein target today?'
                )

                mock_client_cls.assert_called_once_with(api_key='test_key_12345')
                mock_models.generate_content.assert_called_once()
                call_kwargs = mock_models.generate_content.call_args.kwargs
                assert call_kwargs['model'] == 'gemini-3.8-flash'
                assert 'system_instruction' in call_kwargs['config']
                assert result['reply'] == mock_response.text
