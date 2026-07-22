/* ==========================================================
Exact file location: app/static/widget/horseshoe-widget.js
Horseshoe Tavern AI
Phase 1 Part 1.6
Global persistent chatbot widget
========================================================== */

(() => {
    "use strict";

    // ======================================================
    // SECTION 01 - DUPLICATE INITIALIZATION PROTECTION
    // ======================================================

    if (window.HorseshoeAIWidgetInitialized === true) {
        return;
    }

    window.HorseshoeAIWidgetInitialized = true;

    // ======================================================
    // SECTION 02 - CONSTANTS
    // ======================================================

    const WIDGET_VERSION = "1.0.0";

    const STORAGE_PREFIX = "horseshoe_ai_";

    const STORAGE_KEYS = Object.freeze({
        sessionId: `${STORAGE_PREFIX}session_id`,
        conversationId: `${STORAGE_PREFIX}conversation_id`,
        state: `${STORAGE_PREFIX}widget_state`,
        size: `${STORAGE_PREFIX}widget_size`,
        unreadCount: `${STORAGE_PREFIX}unread_count`,
        messages: `${STORAGE_PREFIX}messages`,
        lastPageUrl: `${STORAGE_PREFIX}last_page_url`,
        lastActiveAt: `${STORAGE_PREFIX}last_active_at`,
        privateEventDraft: `${STORAGE_PREFIX}private_event_draft`
    });

    const WIDGET_STATES = Object.freeze({
        collapsed: "collapsed",
        open: "open",
        minimized: "minimized"
    });

    const WIDGET_SIZES = Object.freeze({
        compact: "compact",
        expanded: "expanded",
        fullscreen: "fullscreen"
    });

    const MAX_LOCAL_MESSAGES = 50;
    const MAX_MESSAGE_LENGTH = 3000;
    const REQUEST_TIMEOUT_MS = 25000;
    const RETRY_BASE_DELAY_MS = 1200;
    const MAX_RETRY_ATTEMPTS = 3;

    // ======================================================
    // SECTION 03 - SCRIPT CONFIGURATION
    // ======================================================

    const currentScript = document.currentScript;

    const scriptSource = currentScript?.src || "";
    const inferredBaseUrl = scriptSource
        ? new URL(scriptSource, window.location.href).origin
        : window.location.origin;

    const configuration = Object.freeze({
        baseUrl:
            currentScript?.dataset.baseUrl?.trim() ||
            inferredBaseUrl,

        business:
            currentScript?.dataset.business?.trim() ||
            "horseshoe-tavern",

        position:
            currentScript?.dataset.position?.trim() ||
            "bottom-right",

        defaultSize:
            normalizeWidgetSize(
                currentScript?.dataset.defaultSize || "compact"
            ),

        defaultState:
            normalizeWidgetState(
                currentScript?.dataset.defaultState || "collapsed"
            ),

        chatEndpoint:
            currentScript?.dataset.chatEndpoint?.trim() ||
            "/api/chat",

        sessionEndpoint:
            currentScript?.dataset.sessionEndpoint?.trim() ||
            "/api/widget/session",

        restoreEndpoint:
            currentScript?.dataset.restoreEndpoint?.trim() ||
            "/api/widget/conversation",

        stylesheetUrl:
            currentScript?.dataset.stylesheetUrl?.trim() ||
            `${inferredBaseUrl}/static/widget/horseshoe-widget.css`,

        title:
            currentScript?.dataset.title?.trim() ||
            "Horseshoe Concierge",

        launcherLabel:
            currentScript?.dataset.launcherLabel?.trim() ||
            "Ask Horseshoe",

        welcomeMessage:
            currentScript?.dataset.welcomeMessage?.trim() ||
            (
                "Welcome to Horseshoe Tavern. I can help with hours, " +
                "menus, events, parking, ordering, sports, and private parties."
            ),

        persistenceEnabled:
            currentScript?.dataset.persistence !== "false",

        apiEnabled:
            currentScript?.dataset.apiEnabled !== "false",

        debug:
            currentScript?.dataset.debug === "true"
    });

    // ======================================================
    // SECTION 04 - STATE
    // ======================================================

    const state = {
        widgetState: readStoredState(),
        widgetSize: readStoredSize(),
        sessionId: readOrCreateSessionId(),
        conversationId: readStorage(STORAGE_KEYS.conversationId),
        unreadCount: readStoredInteger(
            STORAGE_KEYS.unreadCount,
            0
        ),
        messages: readStoredMessages(),
        isSending: false,
        isOnline: navigator.onLine,
        retryAttempt: 0,
        previousSize: WIDGET_SIZES.compact,
        pageContext: collectPageContext(),
        initializedAt: new Date().toISOString(),
        lastActiveAt:
            readStorage(STORAGE_KEYS.lastActiveAt) ||
            new Date().toISOString()
    };

    // ======================================================
    // SECTION 05 - DOM REFERENCES
    // ======================================================

    let hostElement = null;
    let shadowRoot = null;

    const elements = {
        wrapper: null,
        launcher: null,
        launcherLabel: null,
        launcherBadge: null,
        panel: null,
        header: null,
        connectionStatus: null,
        messages: null,
        quickActions: null,
        form: null,
        input: null,
        sendButton: null,
        minimizeButton: null,
        resizeButton: null,
        fullscreenButton: null,
        closeButton: null,
        newConversationButton: null,
        liveRegion: null
    };

    // ======================================================
    // SECTION 06 - NORMALIZATION HELPERS
    // ======================================================

    function normalizeWidgetState(value) {
        const candidate = String(value || "").toLowerCase();

        return Object.values(WIDGET_STATES).includes(candidate)
            ? candidate
            : WIDGET_STATES.collapsed;
    }

    function normalizeWidgetSize(value) {
        const candidate = String(value || "").toLowerCase();

        return Object.values(WIDGET_SIZES).includes(candidate)
            ? candidate
            : WIDGET_SIZES.compact;
    }

    function sanitizeText(value) {
        return String(value ?? "")
            .replace(/\u0000/g, "")
            .trim();
    }

    function clampInteger(value, minimum, maximum) {
        const parsed = Number.parseInt(value, 10);

        if (!Number.isFinite(parsed)) {
            return minimum;
        }

        return Math.min(maximum, Math.max(minimum, parsed));
    }

    // ======================================================
    // SECTION 07 - STORAGE HELPERS
    // ======================================================

    function readStorage(key) {
        if (!configuration.persistenceEnabled) {
            return null;
        }

        try {
            return window.localStorage.getItem(key);
        } catch (error) {
            logDebug("Unable to read localStorage.", error);
            return null;
        }
    }

    function writeStorage(key, value) {
        if (!configuration.persistenceEnabled) {
            return;
        }

        try {
            window.localStorage.setItem(key, String(value));
        } catch (error) {
            logDebug("Unable to write localStorage.", error);
        }
    }

    function removeStorage(key) {
        if (!configuration.persistenceEnabled) {
            return;
        }

        try {
            window.localStorage.removeItem(key);
        } catch (error) {
            logDebug("Unable to remove localStorage value.", error);
        }
    }

    function readStoredState() {
        return normalizeWidgetState(
            readStorage(STORAGE_KEYS.state) ||
            configuration.defaultState
        );
    }

    function readStoredSize() {
        return normalizeWidgetSize(
            readStorage(STORAGE_KEYS.size) ||
            configuration.defaultSize
        );
    }

    function readStoredInteger(key, fallback) {
        const value = readStorage(key);

        if (value === null) {
            return fallback;
        }

        return clampInteger(value, 0, 999);
    }

    function readStoredMessages() {
        const serialized = readStorage(STORAGE_KEYS.messages);

        if (!serialized) {
            return [];
        }

        try {
            const parsed = JSON.parse(serialized);

            if (!Array.isArray(parsed)) {
                return [];
            }

            return parsed
                .filter(isValidStoredMessage)
                .slice(-MAX_LOCAL_MESSAGES);
        } catch (error) {
            logDebug("Unable to parse stored messages.", error);
            return [];
        }
    }

    function isValidStoredMessage(message) {
        return Boolean(
            message &&
            typeof message === "object" &&
            ["assistant", "user", "system"].includes(message.role) &&
            typeof message.text === "string"
        );
    }

    function persistMessages() {
        const limitedMessages = state.messages.slice(
            -MAX_LOCAL_MESSAGES
        );

        writeStorage(
            STORAGE_KEYS.messages,
            JSON.stringify(limitedMessages)
        );
    }

    function readOrCreateSessionId() {
        const existing = readStorage(STORAGE_KEYS.sessionId);

        if (existing) {
            return existing;
        }

        const created = createIdentifier("session");

        writeStorage(STORAGE_KEYS.sessionId, created);

        return created;
    }

    function persistState() {
        writeStorage(
            STORAGE_KEYS.state,
            state.widgetState
        );

        writeStorage(
            STORAGE_KEYS.size,
            state.widgetSize
        );

        writeStorage(
            STORAGE_KEYS.unreadCount,
            state.unreadCount
        );

        writeStorage(
            STORAGE_KEYS.lastActiveAt,
            state.lastActiveAt
        );

        writeStorage(
            STORAGE_KEYS.lastPageUrl,
            window.location.href
        );

        if (state.conversationId) {
            writeStorage(
                STORAGE_KEYS.conversationId,
                state.conversationId
            );
        }
    }

    // ======================================================
    // SECTION 08 - IDENTIFIERS
    // ======================================================

    function createIdentifier(prefix) {
        if (
            window.crypto &&
            typeof window.crypto.randomUUID === "function"
        ) {
            return `${prefix}_${window.crypto.randomUUID()}`;
        }

        const randomPart = Math.random()
            .toString(36)
            .slice(2);

        const timestampPart = Date.now().toString(36);

        return `${prefix}_${timestampPart}_${randomPart}`;
    }

    // ======================================================
    // SECTION 09 - PAGE CONTEXT
    // ======================================================

    function collectPageContext() {
        return {
            url: window.location.href,
            path: window.location.pathname,
            title: document.title || "",
            category: inferPageCategory(),
            referrer: document.referrer || "",
            language:
                document.documentElement.lang ||
                navigator.language ||
                "en-US",
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            timestamp: new Date().toISOString()
        };
    }

    function inferPageCategory() {
        const path = window.location.pathname.toLowerCase();

        if (path.includes("private")) {
            return "private_events";
        }

        if (path.includes("event")) {
            return "events";
        }

        if (path.includes("menu")) {
            return "menu";
        }

        if (path.includes("special")) {
            return "specials";
        }

        if (path.includes("contact")) {
            return "contact";
        }

        if (path.includes("order")) {
            return "ordering";
        }

        return "home";
    }

    // ======================================================
    // SECTION 10 - INITIALIZATION
    // ======================================================

    function initialize() {
        if (!document.body) {
            window.addEventListener(
                "DOMContentLoaded",
                initialize,
                { once: true }
            );
            return;
        }

        createHost();
        loadStylesheet();
        createInterface();
        bindEvents();
        restoreInterfaceState();
        renderMessages();
        updateConnectionStatus();

        if (state.messages.length === 0) {
            addMessage({
                role: "assistant",
                text: configuration.welcomeMessage,
                persist: true,
                announce: false
            });
        }

        initializeServerSession();
        exposePublicApi();

        persistState();

        logDebug(
            "Widget initialized.",
            {
                version: WIDGET_VERSION,
                state
            }
        );
    }

    function createHost() {
        hostElement = document.createElement(
            "div"
        );

        hostElement.id = "horseshoe-ai-widget-host";
        hostElement.dataset.version = WIDGET_VERSION;

        shadowRoot = hostElement.attachShadow({
            mode: "open"
        });

        document.body.appendChild(hostElement);
    }

    function loadStylesheet() {
        const stylesheet = document.createElement(
            "link"
        );

        stylesheet.rel = "stylesheet";
        stylesheet.href = configuration.stylesheetUrl;

        shadowRoot.appendChild(stylesheet);
    }

    // ======================================================
    // SECTION 11 - INTERFACE CONSTRUCTION
    // ======================================================

    function createInterface() {
        const container = document.createElement(
            "div"
        );

        container.innerHTML = buildWidgetMarkup();

        elements.wrapper = container.firstElementChild;

        shadowRoot.appendChild(elements.wrapper);

        elements.launcher =
            shadowRoot.querySelector(
                "[data-horseshoe-launcher]"
            );

        elements.launcherLabel =
            shadowRoot.querySelector(
                "[data-horseshoe-launcher-label]"
            );

        elements.launcherBadge =
            shadowRoot.querySelector(
                "[data-horseshoe-unread-badge]"
            );

        elements.panel =
            shadowRoot.querySelector(
                "[data-horseshoe-panel]"
            );

        elements.header =
            shadowRoot.querySelector(
                "[data-horseshoe-header]"
            );

        elements.connectionStatus =
            shadowRoot.querySelector(
                "[data-horseshoe-connection]"
            );

        elements.messages =
            shadowRoot.querySelector(
                "[data-horseshoe-messages]"
            );

        elements.quickActions =
            shadowRoot.querySelector(
                "[data-horseshoe-quick-actions]"
            );

        elements.form =
            shadowRoot.querySelector(
                "[data-horseshoe-form]"
            );

        elements.input =
            shadowRoot.querySelector(
                "[data-horseshoe-input]"
            );

        elements.sendButton =
            shadowRoot.querySelector(
                "[data-horseshoe-send]"
            );

        elements.minimizeButton =
            shadowRoot.querySelector(
                "[data-action='minimize']"
            );

        elements.resizeButton =
            shadowRoot.querySelector(
                "[data-action='resize']"
            );

        elements.fullscreenButton =
            shadowRoot.querySelector(
                "[data-action='fullscreen']"
            );

        elements.closeButton =
            shadowRoot.querySelector(
                "[data-action='close']"
            );

        elements.newConversationButton =
            shadowRoot.querySelector(
                "[data-action='new-conversation']"
            );

        elements.liveRegion =
            shadowRoot.querySelector(
                "[data-horseshoe-live-region]"
            );
    }

    function buildWidgetMarkup() {
        return `
            <section
                class="horseshoe-widget"
                data-widget-state="${escapeAttribute(state.widgetState)}"
                data-widget-size="${escapeAttribute(state.widgetSize)}"
                data-widget-position="${escapeAttribute(configuration.position)}"
                aria-label="Horseshoe Tavern chatbot"
            >
                <button
                    class="horseshoe-launcher"
                    type="button"
                    data-horseshoe-launcher
                    aria-label="Open Horseshoe Tavern chatbot"
                    aria-expanded="false"
                >
                    <span
                        class="horseshoe-launcher__icon"
                        aria-hidden="true"
                    >
                        ✦
                    </span>

                    <span
                        class="horseshoe-launcher__label"
                        data-horseshoe-launcher-label
                    >
                        ${escapeHtml(configuration.launcherLabel)}
                    </span>

                    <span
                        class="horseshoe-launcher__badge"
                        data-horseshoe-unread-badge
                        hidden
                    >
                        0
                    </span>
                </button>

                <section
                    class="horseshoe-panel"
                    data-horseshoe-panel
                    role="dialog"
                    aria-modal="false"
                    aria-label="${escapeAttribute(configuration.title)}"
                    aria-hidden="true"
                >
                    <header
                        class="horseshoe-header"
                        data-horseshoe-header
                    >
                        <div class="horseshoe-header__identity">
                            <div
                                class="horseshoe-header__mark"
                                aria-hidden="true"
                            >
                                H
                            </div>

                            <div class="horseshoe-header__titles">
                                <strong class="horseshoe-header__title">
                                    ${escapeHtml(configuration.title)}
                                </strong>

                                <span
                                    class="horseshoe-header__status"
                                    data-horseshoe-connection
                                >
                                    Connecting
                                </span>
                            </div>
                        </div>

                        <div
                            class="horseshoe-header__controls"
                            aria-label="Chat window controls"
                        >
                            <button
                                type="button"
                                class="horseshoe-control-button"
                                data-action="new-conversation"
                                aria-label="Start a new conversation"
                                title="New conversation"
                            >
                                ↻
                            </button>

                            <button
                                type="button"
                                class="horseshoe-control-button"
                                data-action="resize"
                                aria-label="Change chatbot size"
                                title="Change size"
                            >
                                ◫
                            </button>

                            <button
                                type="button"
                                class="horseshoe-control-button"
                                data-action="fullscreen"
                                aria-label="Open chatbot full screen"
                                title="Full screen"
                            >
                                ⛶
                            </button>

                            <button
                                type="button"
                                class="horseshoe-control-button"
                                data-action="minimize"
                                aria-label="Minimize chatbot"
                                title="Minimize"
                            >
                                —
                            </button>

                            <button
                                type="button"
                                class="horseshoe-control-button"
                                data-action="close"
                                aria-label="Collapse chatbot"
                                title="Close"
                            >
                                ×
                            </button>
                        </div>
                    </header>

                    <main
                        class="horseshoe-conversation"
                        data-horseshoe-messages
                        aria-live="polite"
                        aria-relevant="additions"
                    ></main>

                    <nav
                        class="horseshoe-quick-actions"
                        data-horseshoe-quick-actions
                        aria-label="Popular questions"
                    >
                        ${buildQuickActionButtons()}
                    </nav>

                    <form
                        class="horseshoe-composer"
                        data-horseshoe-form
                    >
                        <label
                            class="horseshoe-visually-hidden"
                            for="horseshoe-ai-input"
                        >
                            Ask Horseshoe Tavern a question
                        </label>

                        <textarea
                            id="horseshoe-ai-input"
                            class="horseshoe-composer__input"
                            data-horseshoe-input
                            rows="1"
                            maxlength="${MAX_MESSAGE_LENGTH}"
                            placeholder="Ask about hours, menus, events, parking..."
                            autocomplete="off"
                            enterkeyhint="send"
                        ></textarea>

                        <button
                            class="horseshoe-composer__send"
                            data-horseshoe-send
                            type="submit"
                            aria-label="Send message"
                        >
                            <span aria-hidden="true">➤</span>
                        </button>
                    </form>

                    <footer class="horseshoe-footer">
                        <span>
                            Powered by Horseshoe Tavern AI
                        </span>

                        <span>
                            Conversation continues across pages
                        </span>
                    </footer>
                </section>

                <div
                    class="horseshoe-visually-hidden"
                    data-horseshoe-live-region
                    aria-live="assertive"
                    aria-atomic="true"
                ></div>
            </section>
        `;
    }

    function buildQuickActionButtons() {
        const actions = [
            {
                label: "Tonight",
                message: "What is happening tonight?"
            },
            {
                label: "Menu",
                message: "Show me the menu."
            },
            {
                label: "Hours",
                message: "What are today's hours?"
            },
            {
                label: "Events",
                message: "What events are coming up?"
            },
            {
                label: "Private party",
                message: "I want to plan a private event."
            },
            {
                label: "Parking",
                message: "Where can I park?"
            }
        ];

        return actions
            .map((action) => `
                <button
                    type="button"
                    class="horseshoe-quick-action"
                    data-quick-message="${escapeAttribute(action.message)}"
                >
                    ${escapeHtml(action.label)}
                </button>
            `)
            .join("");
    }

    // ======================================================
    // SECTION 12 - EVENT BINDING
    // ======================================================

    function bindEvents() {
        elements.launcher.addEventListener(
            "click",
            openWidget
        );

        elements.closeButton.addEventListener(
            "click",
            collapseWidget
        );

        elements.minimizeButton.addEventListener(
            "click",
            minimizeWidget
        );

        elements.resizeButton.addEventListener(
            "click",
            cycleWidgetSize
        );

        elements.fullscreenButton.addEventListener(
            "click",
            toggleFullscreen
        );

        elements.newConversationButton.addEventListener(
            "click",
            requestNewConversation
        );

        elements.form.addEventListener(
            "submit",
            handleFormSubmit
        );

        elements.input.addEventListener(
            "keydown",
            handleInputKeyDown
        );

        elements.input.addEventListener(
            "input",
            autoResizeInput
        );

        elements.quickActions.addEventListener(
            "click",
            handleQuickAction
        );

        window.addEventListener(
            "online",
            handleOnline
        );

        window.addEventListener(
            "offline",
            handleOffline
        );

        window.addEventListener(
            "resize",
            handleViewportChange
        );

        window.addEventListener(
            "pagehide",
            persistState
        );

        document.addEventListener(
            "visibilitychange",
            handleVisibilityChange
        );

        document.addEventListener(
            "keydown",
            handleGlobalKeyboard
        );
    }

    // ======================================================
    // SECTION 13 - INTERFACE STATE
    // ======================================================

    function restoreInterfaceState() {
        applyWidgetState();
        applyWidgetSize();
        updateUnreadBadge();
    }

    function applyWidgetState() {
        elements.wrapper.dataset.widgetState =
            state.widgetState;

        const isOpen =
            state.widgetState === WIDGET_STATES.open;

        elements.panel.setAttribute(
            "aria-hidden",
            String(!isOpen)
        );

        elements.launcher.setAttribute(
            "aria-expanded",
            String(isOpen)
        );

        if (isOpen) {
            resetUnreadCount();

            window.setTimeout(() => {
                elements.input.focus({
                    preventScroll: true
                });
            }, 100);
        }

        persistState();
    }

    function applyWidgetSize() {
        elements.wrapper.dataset.widgetSize =
            state.widgetSize;

        elements.fullscreenButton.setAttribute(
            "aria-label",
            state.widgetSize === WIDGET_SIZES.fullscreen
                ? "Exit chatbot full screen"
                : "Open chatbot full screen"
        );

        persistState();
    }

    function openWidget() {
        state.widgetState = WIDGET_STATES.open;
        state.lastActiveAt = new Date().toISOString();

        applyWidgetState();
        scrollMessagesToBottom();
    }

    function collapseWidget() {
        state.widgetState = WIDGET_STATES.collapsed;
        state.lastActiveAt = new Date().toISOString();

        if (state.widgetSize === WIDGET_SIZES.fullscreen) {
            state.widgetSize = state.previousSize;
            applyWidgetSize();
        }

        applyWidgetState();
        unlockDocumentScroll();
    }

    function minimizeWidget() {
        state.widgetState = WIDGET_STATES.minimized;
        state.lastActiveAt = new Date().toISOString();

        applyWidgetState();
        unlockDocumentScroll();
    }

    function cycleWidgetSize() {
        if (
            state.widgetSize === WIDGET_SIZES.fullscreen
        ) {
            state.widgetSize =
                state.previousSize ||
                WIDGET_SIZES.compact;
        } else if (
            state.widgetSize === WIDGET_SIZES.compact
        ) {
            state.widgetSize = WIDGET_SIZES.expanded;
        } else {
            state.widgetSize = WIDGET_SIZES.compact;
        }

        applyWidgetSize();
        scrollMessagesToBottom();
    }

    function toggleFullscreen() {
        if (
            state.widgetSize === WIDGET_SIZES.fullscreen
        ) {
            state.widgetSize =
                state.previousSize ||
                WIDGET_SIZES.compact;

            unlockDocumentScroll();
        } else {
            state.previousSize = state.widgetSize;
            state.widgetSize = WIDGET_SIZES.fullscreen;

            lockDocumentScroll();
        }

        state.widgetState = WIDGET_STATES.open;

        applyWidgetState();
        applyWidgetSize();
        scrollMessagesToBottom();
    }

    function lockDocumentScroll() {
        document.documentElement.classList.add(
            "horseshoe-ai-scroll-locked"
        );

        document.body.classList.add(
            "horseshoe-ai-scroll-locked"
        );
    }

    function unlockDocumentScroll() {
        document.documentElement.classList.remove(
            "horseshoe-ai-scroll-locked"
        );

        document.body.classList.remove(
            "horseshoe-ai-scroll-locked"
        );
    }

    // ======================================================
    // SECTION 14 - MESSAGE RENDERING
    // ======================================================

    function renderMessages() {
        elements.messages.innerHTML = "";

        state.messages.forEach((message) => {
            elements.messages.appendChild(
                createMessageElement(message)
            );
        });

        scrollMessagesToBottom(false);
    }

    function createMessageElement(message) {
        const article = document.createElement(
            "article"
        );

        article.className =
            `horseshoe-message horseshoe-message--${message.role}`;

        article.dataset.messageId =
            message.id || "";

        const bubble = document.createElement(
            "div"
        );

        bubble.className =
            "horseshoe-message__bubble";

        const text = document.createElement(
            "p"
        );

        text.className =
            "horseshoe-message__text";

        text.textContent = message.text;

        bubble.appendChild(text);

        if (
            message.role === "assistant" &&
            Array.isArray(message.actions) &&
            message.actions.length > 0
        ) {
            bubble.appendChild(
                createMessageActions(message.actions)
            );
        }

        article.appendChild(bubble);

        const metadata = document.createElement(
            "span"
        );

        metadata.className =
            "horseshoe-message__metadata";

        metadata.textContent =
            formatMessageTime(message.createdAt);

        article.appendChild(metadata);

        return article;
    }

    function createMessageActions(actions) {
        const container = document.createElement(
            "div"
        );

        container.className =
            "horseshoe-message-actions";

        actions.forEach((action) => {
            if (
                !action ||
                typeof action !== "object" ||
                !action.label
            ) {
                return;
            }

            const button = document.createElement(
                action.url ? "a" : "button"
            );

            button.className =
                "horseshoe-message-action";

            button.textContent =
                String(action.label);

            if (action.url) {
                button.href = String(action.url);
                button.target =
                    action.target || "_blank";

                button.rel =
                    "noopener noreferrer";
            } else {
                button.type = "button";

                button.addEventListener(
                    "click",
                    () => {
                        sendMessage(
                            action.message ||
                            action.label
                        );
                    }
                );
            }

            container.appendChild(button);
        });

        return container;
    }

    function addMessage({
        role,
        text,
        actions = [],
        persist = true,
        announce = true
    }) {
        const cleanText = sanitizeText(text);

        if (!cleanText) {
            return null;
        }

        const message = {
            id: createIdentifier("message"),
            role,
            text: cleanText,
            actions: Array.isArray(actions)
                ? actions
                : [],
            createdAt: new Date().toISOString(),
            pageUrl: window.location.href
        };

        state.messages.push(message);

        if (
            state.messages.length >
            MAX_LOCAL_MESSAGES
        ) {
            state.messages =
                state.messages.slice(
                    -MAX_LOCAL_MESSAGES
                );
        }

        elements.messages.appendChild(
            createMessageElement(message)
        );

        if (persist) {
            persistMessages();
        }

        if (
            role === "assistant" &&
            state.widgetState !== WIDGET_STATES.open
        ) {
            incrementUnreadCount();
        }

        if (announce) {
            announceMessage(cleanText);
        }

        scrollMessagesToBottom();

        return message;
    }

    function addTypingIndicator() {
        removeTypingIndicator();

        const indicator = document.createElement(
            "article"
        );

        indicator.className =
            "horseshoe-message horseshoe-message--assistant";

        indicator.dataset.typingIndicator =
            "true";

        indicator.innerHTML = `
            <div class="horseshoe-message__bubble">
                <div
                    class="horseshoe-typing"
                    aria-label="Horseshoe is typing"
                >
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        elements.messages.appendChild(indicator);
        scrollMessagesToBottom();
    }

    function removeTypingIndicator() {
        const indicator =
            elements.messages.querySelector(
                "[data-typing-indicator='true']"
            );

        indicator?.remove();
    }

    function formatMessageTime(value) {
        try {
            return new Intl.DateTimeFormat(
                undefined,
                {
                    hour: "numeric",
                    minute: "2-digit"
                }
            ).format(new Date(value));
        } catch {
            return "";
        }
    }

    function scrollMessagesToBottom(smooth = true) {
        window.requestAnimationFrame(() => {
            elements.messages.scrollTo({
                top:
                    elements.messages.scrollHeight,
                behavior:
                    smooth ? "smooth" : "auto"
            });
        });
    }

    // ======================================================
    // SECTION 15 - UNREAD MANAGEMENT
    // ======================================================

    function incrementUnreadCount() {
        state.unreadCount = Math.min(
            99,
            state.unreadCount + 1
        );

        updateUnreadBadge();
        persistState();
    }

    function resetUnreadCount() {
        state.unreadCount = 0;

        updateUnreadBadge();
        persistState();
    }

    function updateUnreadBadge() {
        const count = state.unreadCount;

        elements.launcherBadge.textContent =
            count > 99 ? "99+" : String(count);

        elements.launcherBadge.hidden =
            count <= 0;
    }

    // ======================================================
    // SECTION 16 - INPUT HANDLING
    // ======================================================

    function handleFormSubmit(event) {
        event.preventDefault();

        const message = sanitizeText(
            elements.input.value
        );

        if (!message || state.isSending) {
            return;
        }

        elements.input.value = "";
        autoResizeInput();

        sendMessage(message);
    }

    function handleInputKeyDown(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            elements.form.requestSubmit();
        }
    }

    function autoResizeInput() {
        elements.input.style.height = "auto";

        elements.input.style.height =
            `${Math.min(
                elements.input.scrollHeight,
                128
            )}px`;
    }

    function handleQuickAction(event) {
        const button = event.target.closest(
            "[data-quick-message]"
        );

        if (!button) {
            return;
        }

        const message =
            button.dataset.quickMessage;

        sendMessage(message);
    }

    // ======================================================
    // SECTION 17 - MESSAGE TRANSPORT
    // ======================================================

    async function sendMessage(messageText) {
        const cleanMessage = sanitizeText(
            messageText
        ).slice(0, MAX_MESSAGE_LENGTH);

        if (!cleanMessage || state.isSending) {
            return;
        }

        openWidget();

        addMessage({
            role: "user",
            text: cleanMessage
        });

        state.isSending = true;
        state.lastActiveAt =
            new Date().toISOString();

        setComposerBusy(true);
        addTypingIndicator();

        try {
            const response = configuration.apiEnabled
                ? await requestChatResponse(cleanMessage)
                : buildLocalFallbackResponse(cleanMessage);

            removeTypingIndicator();

            if (response.conversationId) {
                state.conversationId =
                    response.conversationId;
            }

            addMessage({
                role: "assistant",
                text: response.message,
                actions: response.actions || []
            });

            state.retryAttempt = 0;
            persistState();
        } catch (error) {
            removeTypingIndicator();

            addMessage({
                role: "assistant",
                text: buildConnectionFailureMessage(),
                actions: [
                    {
                        label: "Try again",
                        message: cleanMessage
                    }
                ]
            });

            logDebug(
                "Chat request failed.",
                error
            );
        } finally {
            state.isSending = false;
            setComposerBusy(false);

            elements.input.focus({
                preventScroll: true
            });
        }
    }

    async function requestChatResponse(message) {
        const controller =
            new AbortController();

        const timeoutId = window.setTimeout(
            () => controller.abort(),
            REQUEST_TIMEOUT_MS
        );

        const payload = {
            session_id: state.sessionId,
            conversation_id:
                state.conversationId,
            message,
            business_slug:
                configuration.business,
            page_context:
                collectPageContext(),
            widget_context: {
                version: WIDGET_VERSION,
                state: state.widgetState,
                size: state.widgetSize,
                previous_page_url:
                    readStorage(
                        STORAGE_KEYS.lastPageUrl
                    ),
                last_active_at:
                    state.lastActiveAt
            }
        };

        try {
            const response = await fetch(
                `${configuration.baseUrl}${configuration.chatEndpoint}`,
                {
                    method: "POST",
                    mode: "cors",
                    credentials: "omit",
                    headers: {
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "X-Horseshoe-Session":
                            state.sessionId,
                        "X-Horseshoe-Widget-Version":
                            WIDGET_VERSION
                    },
                    body: JSON.stringify(payload),
                    signal: controller.signal
                }
            );

            if (!response.ok) {
                throw new Error(
                    `Chat request failed with status ${response.status}.`
                );
            }

            const data = await response.json();

            return normalizeChatResponse(data);
        } finally {
            window.clearTimeout(timeoutId);
        }
    }

    function normalizeChatResponse(data) {
        const message =
            sanitizeText(
                data?.message ||
                data?.response ||
                data?.answer
            );

        if (!message) {
            throw new Error(
                "The chat service returned an empty response."
            );
        }

        return {
            message,
            conversationId:
                sanitizeText(
                    data?.conversation_id ||
                    data?.conversationId
                ) || null,
            actions:
                Array.isArray(data?.actions)
                    ? data.actions
                    : []
        };
    }

    function buildLocalFallbackResponse(message) {
        const normalized =
            message.toLowerCase();

        if (
            normalized.includes("hour") ||
            normalized.includes("open") ||
            normalized.includes("close")
        ) {
            return {
                message:
                    "I am currently reconnecting to the Horseshoe service. " +
                    "Please use the tavern website or call the business to " +
                    "confirm today's hours.",
                actions: []
            };
        }

        return {
            message:
                "I saved your message locally, but I cannot reach the " +
                "Horseshoe service right now. Please try again shortly.",
            actions: []
        };
    }

    function buildConnectionFailureMessage() {
        if (!navigator.onLine) {
            return (
                "You appear to be offline. Your conversation is still " +
                "saved on this device. Reconnect and try again."
            );
        }

        return (
            "I could not reach the Horseshoe service just now. " +
            "Your conversation is still saved. Please try again."
        );
    }

    function setComposerBusy(isBusy) {
        elements.input.disabled = isBusy;
        elements.sendButton.disabled = isBusy;

        elements.sendButton.classList.toggle(
            "is-loading",
            isBusy
        );

        elements.panel.setAttribute(
            "aria-busy",
            String(isBusy)
        );
    }

    // ======================================================
    // SECTION 18 - SERVER SESSION
    // ======================================================

    async function initializeServerSession() {
        if (!configuration.apiEnabled) {
            return;
        }

        try {
            const response = await fetch(
                `${configuration.baseUrl}${configuration.sessionEndpoint}`,
                {
                    method: "POST",
                    mode: "cors",
                    credentials: "omit",
                    headers: {
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        session_id: state.sessionId,
                        conversation_id:
                            state.conversationId,
                        business_slug:
                            configuration.business,
                        page_context:
                            collectPageContext(),
                        widget_version:
                            WIDGET_VERSION
                    })
                }
            );

            if (!response.ok) {
                throw new Error(
                    `Session initialization returned ${response.status}.`
                );
            }

            const data = await response.json();

            const conversationId =
                sanitizeText(
                    data?.conversation_id ||
                    data?.conversationId
                );

            if (conversationId) {
                state.conversationId =
                    conversationId;

                writeStorage(
                    STORAGE_KEYS.conversationId,
                    conversationId
                );
            }

            if (
                Array.isArray(data?.messages) &&
                data.messages.length > 0 &&
                state.messages.length <= 1
            ) {
                restoreServerMessages(
                    data.messages
                );
            }

            state.isOnline = true;
            updateConnectionStatus();
        } catch (error) {
            state.isOnline = navigator.onLine;
            updateConnectionStatus();

            logDebug(
                "Session initialization deferred.",
                error
            );
        }
    }

    function restoreServerMessages(messages) {
        const normalized = messages
            .map((message) => ({
                id:
                    message.id ||
                    createIdentifier("message"),
                role:
                    ["assistant", "user", "system"].includes(
                        message.role
                    )
                        ? message.role
                        : "assistant",
                text:
                    sanitizeText(
                        message.text ||
                        message.message
                    ),
                actions:
                    Array.isArray(message.actions)
                        ? message.actions
                        : [],
                createdAt:
                    message.created_at ||
                    message.createdAt ||
                    new Date().toISOString(),
                pageUrl:
                    message.page_url ||
                    message.pageUrl ||
                    ""
            }))
            .filter((message) => message.text);

        if (normalized.length === 0) {
            return;
        }

        state.messages = normalized.slice(
            -MAX_LOCAL_MESSAGES
        );

        persistMessages();
        renderMessages();
    }

    // ======================================================
    // SECTION 19 - NEW CONVERSATION
    // ======================================================

    function requestNewConversation() {
        const confirmed = window.confirm(
            "Start a new Horseshoe conversation? " +
            "The current conversation will remain stored on the server."
        );

        if (!confirmed) {
            return;
        }

        state.sessionId =
            createIdentifier("session");

        state.conversationId = null;
        state.unreadCount = 0;
        state.messages = [];

        removeStorage(
            STORAGE_KEYS.conversationId
        );

        removeStorage(
            STORAGE_KEYS.messages
        );

        removeStorage(
            STORAGE_KEYS.privateEventDraft
        );

        writeStorage(
            STORAGE_KEYS.sessionId,
            state.sessionId
        );

        renderMessages();

        addMessage({
            role: "assistant",
            text: configuration.welcomeMessage
        });

        updateUnreadBadge();
        persistState();
        initializeServerSession();
    }

    // ======================================================
    // SECTION 20 - CONNECTIVITY
    // ======================================================

    function handleOnline() {
        state.isOnline = true;
        state.retryAttempt = 0;

        updateConnectionStatus();
        initializeServerSession();
    }

    function handleOffline() {
        state.isOnline = false;
        updateConnectionStatus();
    }

    function updateConnectionStatus() {
        if (!elements.connectionStatus) {
            return;
        }

        elements.connectionStatus.textContent =
            state.isOnline
                ? "Online"
                : "Offline";

        elements.connectionStatus.dataset.status =
            state.isOnline
                ? "online"
                : "offline";
    }

    // ======================================================
    // SECTION 21 - GLOBAL EVENTS
    // ======================================================

    function handleViewportChange() {
        state.pageContext =
            collectPageContext();

        if (
            window.innerWidth <= 640 &&
            state.widgetSize === WIDGET_SIZES.expanded
        ) {
            state.widgetSize =
                WIDGET_SIZES.fullscreen;

            applyWidgetSize();
        }
    }

    function handleVisibilityChange() {
        if (!document.hidden) {
            state.pageContext =
                collectPageContext();

            state.lastActiveAt =
                new Date().toISOString();

            if (
                state.widgetState ===
                WIDGET_STATES.open
            ) {
                resetUnreadCount();
            }
        }

        persistState();
    }

    function handleGlobalKeyboard(event) {
        if (event.key === "Escape") {
            if (
                state.widgetSize ===
                WIDGET_SIZES.fullscreen
            ) {
                toggleFullscreen();
                return;
            }

            if (
                state.widgetState ===
                WIDGET_STATES.open
            ) {
                collapseWidget();
            }
        }
    }

    // ======================================================
    // SECTION 22 - ACCESSIBILITY
    // ======================================================

    function announceMessage(message) {
        if (!elements.liveRegion) {
            return;
        }

        elements.liveRegion.textContent = "";

        window.setTimeout(() => {
            elements.liveRegion.textContent =
                message;
        }, 50);
    }

    // ======================================================
    // SECTION 23 - HTML ESCAPING
    // ======================================================

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function escapeAttribute(value) {
        return escapeHtml(value);
    }

    // ======================================================
    // SECTION 24 - PUBLIC API
    // ======================================================

    function exposePublicApi() {
        window.HorseshoeAI = Object.freeze({
            version: WIDGET_VERSION,

            open: openWidget,

            close: collapseWidget,

            minimize: minimizeWidget,

            toggleFullscreen,

            setSize(size) {
                state.widgetSize =
                    normalizeWidgetSize(size);

                applyWidgetSize();
            },

            send(message) {
                sendMessage(message);
            },

            newConversation:
                requestNewConversation,

            getState() {
                return {
                    widgetState:
                        state.widgetState,
                    widgetSize:
                        state.widgetSize,
                    sessionId:
                        state.sessionId,
                    conversationId:
                        state.conversationId,
                    unreadCount:
                        state.unreadCount,
                    pageContext:
                        collectPageContext()
                };
            },

            refreshPageContext() {
                state.pageContext =
                    collectPageContext();

                return state.pageContext;
            }
        });
    }

    // ======================================================
    // SECTION 25 - DEBUGGING
    // ======================================================

    function logDebug(message, metadata = null) {
        if (!configuration.debug) {
            return;
        }

        console.debug(
            `[HorseshoeAI ${WIDGET_VERSION}] ${message}`,
            metadata
        );
    }

    // ======================================================
    // SECTION 26 - STARTUP
    // ======================================================

    initialize();
})();
