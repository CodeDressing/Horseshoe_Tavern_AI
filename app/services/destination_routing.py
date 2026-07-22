# ============================================================
# Exact file location: app/services/destination_routing.py
# Horseshoe Tavern AI
# Phase 1 Part 1.38
# Official destination routing, CTAs, and Schema.org metadata
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from app.schemas.chat import ResponseAction, ResponseActionType

DESTINATION_ROUTING_VERSION: Final[str] = "1.0.0"
DESTINATION_ROUTING_PHASE: Final[str] = "Phase 1 Part 1.38"

OFFICIAL_WEBSITE_URL: Final[str] = "https://www.thehorseshoetavern.com/"
OFFICIAL_MENU_URL: Final[str] = "https://www.thehorseshoetavern.com/menu"
OFFICIAL_SPECIALS_URL: Final[str] = "https://www.thehorseshoetavern.com/specials"
OFFICIAL_EVENTS_URL: Final[str] = "https://www.thehorseshoetavern.com/events"
OFFICIAL_GALLERY_URL: Final[str] = "https://www.thehorseshoetavern.com/gallery"
OFFICIAL_PRIVATE_EVENTS_URL: Final[str] = "https://www.thehorseshoetavern.com/private_events"
OFFICIAL_CONTACT_URL: Final[str] = "https://www.thehorseshoetavern.com/contact"
DELIVERY_ORDER_URL: Final[str] = "https://www.chownow.com/order/22979/locations/33522"
PICKUP_ORDER_URL: Final[str] = (
    "https://order.spoton.com/sau-horseshoe-tavern-14533/"
    "morristown-nj/649f110b19dac14bb6791e38"
)
FACEBOOK_URL: Final[str] = "https://www.facebook.com/NJHorseshoe"
INSTAGRAM_URL: Final[str] = "https://www.instagram.com/njhorseshoe"
GENERAL_PHONE_E164: Final[str] = "+19739988447"
GENERAL_EMAIL: Final[str] = "info@thehorseshoetavern.com"
PRIVATE_EVENT_PHONE_E164: Final[str] = "+19732558208"
PRIVATE_EVENT_EMAIL: Final[str] = "events@thehorseshoetavern.com"


class DestinationKey(str, Enum):
    HOME = "home"
    MENU = "menu"
    SPECIALS = "specials"
    EVENTS = "events"
    GALLERY = "gallery"
    PRIVATE_EVENTS = "private_events"
    CONTACT = "contact"
    ORDER = "order"
    DELIVERY = "delivery"
    PICKUP = "pickup"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


@dataclass(frozen=True, slots=True)
class DestinationDefinition:
    key: DestinationKey
    title: str
    description: str
    url: str
    label: str
    analytics_event: str
    phrases: tuple[str, ...]
    action_type: ResponseActionType = ResponseActionType.LINK
    phone_number: str | None = None
    email_address: str | None = None
    provider: str = "official_website"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.value,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "label": self.label,
            "analytics_event": self.analytics_event,
            "provider": self.provider,
        }


@dataclass(frozen=True, slots=True)
class DestinationMatch:
    destination: DestinationDefinition
    confidence: float
    matched_phrases: tuple[str, ...]
    match_method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination.as_dict(),
            "confidence": self.confidence,
            "matched_phrases": list(self.matched_phrases),
            "match_method": self.match_method,
        }


DESTINATIONS: Final[tuple[DestinationDefinition, ...]] = (
    DestinationDefinition(
        DestinationKey.DELIVERY,
        "Horseshoe Tavern Delivery",
        "Place a delivery order through the official ChowNow ordering page.",
        DELIVERY_ORDER_URL,
        "Order Delivery",
        "delivery_order_started",
        ("delivery", "deliver", "order delivery", "chownow"),
        ResponseActionType.ORDERING,
        provider="chownow",
    ),
    DestinationDefinition(
        DestinationKey.PICKUP,
        "Horseshoe Tavern Pickup",
        "Place a pickup or takeout order through the official SpotOn ordering page.",
        PICKUP_ORDER_URL,
        "Order Pickup",
        "pickup_order_started",
        ("pickup", "pick up", "takeout", "take out", "carryout", "spoton"),
        ResponseActionType.ORDERING,
        provider="spoton",
    ),
    DestinationDefinition(
        DestinationKey.PRIVATE_EVENTS,
        "Horseshoe Tavern Private Events",
        "Review private-event information and inquiry options.",
        OFFICIAL_PRIVATE_EVENTS_URL,
        "Plan a Private Event",
        "private_events_opened",
        (
            "private event", "private party", "book a party", "birthday party",
            "corporate event", "holiday party", "event space", "private room",
            "party package", "banquet", "catering",
        ),
        phone_number=PRIVATE_EVENT_PHONE_E164,
        email_address=PRIVATE_EVENT_EMAIL,
    ),
    DestinationDefinition(
        DestinationKey.SPECIALS,
        "Horseshoe Tavern Specials",
        "View current official food and drink specials.",
        OFFICIAL_SPECIALS_URL,
        "View Specials",
        "specials_opened",
        ("special", "specials", "daily specials", "drink specials", "food specials", "happy hour"),
    ),
    DestinationDefinition(
        DestinationKey.EVENTS,
        "Horseshoe Tavern Events",
        "View upcoming official events and entertainment.",
        OFFICIAL_EVENTS_URL,
        "View Events",
        "events_opened",
        ("event", "events", "live music", "trivia", "karaoke", "dj", "watch party", "entertainment"),
    ),
    DestinationDefinition(
        DestinationKey.GALLERY,
        "Horseshoe Tavern Gallery",
        "View official venue and event photographs.",
        OFFICIAL_GALLERY_URL,
        "View Gallery",
        "gallery_opened",
        ("gallery", "photos", "pictures", "images", "what does it look like"),
    ),
    DestinationDefinition(
        DestinationKey.MENU,
        "Horseshoe Tavern Menu",
        "View the official food and drink menu.",
        OFFICIAL_MENU_URL,
        "View Menu",
        "menu_opened",
        ("menu", "food menu", "drink menu", "cocktail menu", "what do you serve", "what food do you have"),
    ),
    DestinationDefinition(
        DestinationKey.FACEBOOK,
        "Horseshoe Tavern Facebook",
        "Open the official Facebook page.",
        FACEBOOK_URL,
        "Open Facebook",
        "facebook_opened",
        ("facebook", "fb"),
        provider="facebook",
    ),
    DestinationDefinition(
        DestinationKey.INSTAGRAM,
        "Horseshoe Tavern Instagram",
        "Open the official Instagram page.",
        INSTAGRAM_URL,
        "Open Instagram",
        "instagram_opened",
        ("instagram", "insta", "ig"),
        provider="instagram",
    ),
    DestinationDefinition(
        DestinationKey.CONTACT,
        "Contact Horseshoe Tavern",
        "Open the official contact page or contact the tavern directly.",
        OFFICIAL_CONTACT_URL,
        "Contact the Tavern",
        "contact_opened",
        ("contact", "call", "phone", "email", "address", "directions", "location", "reservation", "speak to someone"),
        phone_number=GENERAL_PHONE_E164,
        email_address=GENERAL_EMAIL,
    ),
    DestinationDefinition(
        DestinationKey.ORDER,
        "Order from Horseshoe Tavern",
        "Choose delivery or pickup ordering.",
        OFFICIAL_MENU_URL,
        "Order Online",
        "ordering_opened",
        ("order", "order online", "place an order", "online ordering"),
        ResponseActionType.ORDERING,
    ),
    DestinationDefinition(
        DestinationKey.HOME,
        "Official Horseshoe Tavern Website",
        "Open the official Horseshoe Tavern website.",
        OFFICIAL_WEBSITE_URL,
        "Visit Official Website",
        "website_opened",
        ("website", "official website", "homepage", "home page"),
    ),
)

INTENT_MAP: Final[dict[str, DestinationKey]] = {
    "menu_general": DestinationKey.MENU,
    "menu_item_lookup": DestinationKey.MENU,
    "menu_dietary": DestinationKey.MENU,
    "menu_allergen": DestinationKey.MENU,
    "menu_price": DestinationKey.MENU,
    "happy_hour": DestinationKey.SPECIALS,
    "hours_happy_hour": DestinationKey.SPECIALS,
    "events_general": DestinationKey.EVENTS,
    "events_tonight": DestinationKey.EVENTS,
    "live_music": DestinationKey.EVENTS,
    "sports_viewing": DestinationKey.EVENTS,
    "private_event": DestinationKey.PRIVATE_EVENTS,
    "private_event_pricing": DestinationKey.PRIVATE_EVENTS,
    "private_event_availability": DestinationKey.PRIVATE_EVENTS,
    "private_event_contact": DestinationKey.PRIVATE_EVENTS,
    "reservation": DestinationKey.CONTACT,
    "reservation_change": DestinationKey.CONTACT,
    "reservation_cancel": DestinationKey.CONTACT,
    "human_handoff": DestinationKey.CONTACT,
    "complaint": DestinationKey.CONTACT,
}


def _normalize(value: Any) -> str:
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9@.+']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _request_text(nlu_result: Any) -> str:
    values: list[str] = []
    for name in ("original_text", "raw_text", "normalized_text", "corrected_text", "text", "message", "query"):
        value = getattr(nlu_result, name, None)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return _normalize(" ".join(dict.fromkeys(values)))


def _intent_name(nlu_result: Any) -> str:
    intent = getattr(nlu_result, "primary_intent", "")
    value = getattr(intent, "value", intent)
    return _normalize(value).replace(" ", "_")


def _contains(text: str, phrase: str) -> bool:
    p = _normalize(phrase)
    return bool(p and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", text))


def match_destinations(nlu_result: Any, *, maximum_results: int = 4) -> tuple[DestinationMatch, ...]:
    text = _request_text(nlu_result)
    intent_name = _intent_name(nlu_result)
    matches: dict[DestinationKey, DestinationMatch] = {}

    mapped_key = INTENT_MAP.get(intent_name)
    if mapped_key is not None:
        destination = next(item for item in DESTINATIONS if item.key == mapped_key)
        matches[mapped_key] = DestinationMatch(destination, 0.88, (), "intent_map")

    for destination in DESTINATIONS:
        phrases = tuple(phrase for phrase in destination.phrases if _contains(text, phrase))
        if not phrases:
            continue
        confidence = min(0.99, 0.58 + (0.08 * len(phrases)) + (0.04 * max(len(p.split()) for p in phrases)))
        current = matches.get(destination.key)
        if current is None or confidence > current.confidence:
            matches[destination.key] = DestinationMatch(destination, round(confidence, 6), phrases, "verified_phrase")

    if DestinationKey.ORDER in matches:
        for key in (DestinationKey.DELIVERY, DestinationKey.PICKUP):
            if key not in matches:
                destination = next(item for item in DESTINATIONS if item.key == key)
                matches[key] = DestinationMatch(destination, 0.78, (), "ordering_expansion")

    if _contains(text, "social media") or _contains(text, "social pages"):
        for key in (DestinationKey.FACEBOOK, DestinationKey.INSTAGRAM):
            if key not in matches:
                destination = next(item for item in DESTINATIONS if item.key == key)
                matches[key] = DestinationMatch(destination, 0.82, ("social media",), "social_expansion")

    ranked = sorted(matches.values(), key=lambda item: (-item.confidence, item.destination.key.value))
    return tuple(ranked[: max(0, int(maximum_results))])


def _action(destination: DestinationDefinition) -> ResponseAction:
    return ResponseAction(
        action_type=destination.action_type,
        label=destination.label,
        url=destination.url,
        target="_blank",
        analytics_event=destination.analytics_event,
    )


def build_destination_actions(nlu_result: Any, *, maximum_results: int = 4) -> tuple[ResponseAction, ...]:
    matches = match_destinations(nlu_result, maximum_results=maximum_results)
    actions: list[ResponseAction] = [_action(match.destination) for match in matches]

    if matches:
        primary = matches[0].destination
        if primary.phone_number:
            actions.append(ResponseAction(
                action_type=ResponseActionType.PHONE,
                label="Call Private Events" if primary.key == DestinationKey.PRIVATE_EVENTS else "Call the Tavern",
                phone_number=primary.phone_number,
                analytics_event="destination_phone_clicked",
            ))
        if primary.email_address:
            actions.append(ResponseAction(
                action_type=ResponseActionType.EMAIL,
                label="Email Private Events" if primary.key == DestinationKey.PRIVATE_EVENTS else "Email the Tavern",
                email_address=primary.email_address,
                analytics_event="destination_email_clicked",
            ))

    unique: dict[tuple[str, str, str], ResponseAction] = {}
    for action in actions:
        key = (
            str(action.action_type),
            action.label,
            str(action.url or action.phone_number or action.email_address or ""),
        )
        unique.setdefault(key, action)
    return tuple(unique.values())


def destination_fallback_message(nlu_result: Any) -> str | None:
    matches = match_destinations(nlu_result, maximum_results=1)
    if not matches:
        return None
    key = matches[0].destination.key
    messages = {
        DestinationKey.MENU: "You can view the official Horseshoe Tavern menu using the button below.",
        DestinationKey.SPECIALS: "You can view current official specials using the button below.",
        DestinationKey.EVENTS: "You can view current official event listings using the button below.",
        DestinationKey.GALLERY: "You can view official venue photographs using the gallery button below.",
        DestinationKey.PRIVATE_EVENTS: "You can review private-event information and contact options below.",
        DestinationKey.CONTACT: "You can contact Horseshoe Tavern through the official contact options below.",
        DestinationKey.DELIVERY: "You can place a delivery order through the official ChowNow link below.",
        DestinationKey.PICKUP: "You can place a pickup order through the official SpotOn link below.",
        DestinationKey.ORDER: "Choose delivery or pickup below to begin your order.",
        DestinationKey.FACEBOOK: "You can open the official Facebook page below.",
        DestinationKey.INSTAGRAM: "You can open the official Instagram page below.",
        DestinationKey.HOME: "You can open the official Horseshoe Tavern website below.",
    }
    return messages.get(key)


def restaurant_schema() -> dict[str, Any]:
    return {
        "@context": "https://schema.org",
        "@type": "Restaurant",
        "@id": f"{OFFICIAL_WEBSITE_URL}#restaurant",
        "name": "The Horseshoe Tavern",
        "url": OFFICIAL_WEBSITE_URL,
        "telephone": GENERAL_PHONE_E164,
        "email": GENERAL_EMAIL,
        "menu": OFFICIAL_MENU_URL,
        "address": {
            "@type": "PostalAddress",
            "streetAddress": "36 Speedwell Ave",
            "addressLocality": "Morristown",
            "addressRegion": "NJ",
            "postalCode": "07960",
            "addressCountry": "US",
        },
        "sameAs": [FACEBOOK_URL, INSTAGRAM_URL],
        "potentialAction": [
            {"@type": "OrderAction", "name": "Order Delivery", "target": DELIVERY_ORDER_URL},
            {"@type": "OrderAction", "name": "Order Pickup", "target": PICKUP_ORDER_URL},
            {"@type": "ReserveAction", "name": "Plan a Private Event", "target": OFFICIAL_PRIVATE_EVENTS_URL},
        ],
    }


def build_destination_metadata(nlu_result: Any, *, maximum_results: int = 4) -> dict[str, Any]:
    matches = match_destinations(nlu_result, maximum_results=maximum_results)
    return {
        "destination_routing": {
            "matched": bool(matches),
            "match_count": len(matches),
            "matches": [match.as_dict() for match in matches],
            "official_destinations": [match.destination.url for match in matches],
            "service_version": DESTINATION_ROUTING_VERSION,
            "service_phase": DESTINATION_ROUTING_PHASE,
        },
        "restaurant_schema": restaurant_schema(),
    }


def validate_destination_registry() -> dict[str, Any]:
    checks = {
        "destinations_present": len(DESTINATIONS) >= 10,
        "all_https": all(item.url.startswith("https://") for item in DESTINATIONS),
        "unique_keys": len({item.key for item in DESTINATIONS}) == len(DESTINATIONS),
        "restaurant_schema": restaurant_schema().get("@type") == "Restaurant",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "ok" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
    }


__all__ = [
    "DESTINATION_ROUTING_PHASE",
    "DESTINATION_ROUTING_VERSION",
    "DestinationDefinition",
    "DestinationKey",
    "DestinationMatch",
    "build_destination_actions",
    "build_destination_metadata",
    "destination_fallback_message",
    "match_destinations",
    "restaurant_schema",
    "validate_destination_registry",
]


if __name__ == "__main__":
    import json
    report = validate_destination_registry()
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)
