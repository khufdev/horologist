/*
 * Copyright 2022 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.android.horologist.compose.rotaryinput

import android.view.ViewConfiguration
import androidx.compose.animation.core.AnimationState
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.SpringSpec
import androidx.compose.animation.core.animateTo
import androidx.compose.animation.core.copy
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.MutatePriority
import androidx.compose.foundation.OverscrollEffect
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.FlingBehavior
import androidx.compose.foundation.gestures.ScrollableDefaults
import androidx.compose.foundation.gestures.ScrollableState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.input.rotary.onRotaryScrollEvent
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.util.fastSumBy
import androidx.wear.compose.foundation.lazy.ScalingLazyListState
import com.google.android.horologist.annotations.ExperimentalHorologistApi
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.transformLatest
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.min
import kotlin.math.sign

private const val DEBUG = true

/**
 * Debug logging that can be enabled.
 */
private inline fun debugLog(generateMsg: () -> String) {
    if (DEBUG) {
        println("RotaryScroll: ${generateMsg()}")
    }
}

/**
 * A modifier which connects rotary events with scrollable.
 * This modifier supports fling.
 *
 *  Fling algorithm:
 * - A scroll with RSB/ Bezel happens.
 * - If this is a first rotary event after the threshold ( by default 200ms), a new scroll
 * session starts by resetting all necessary parameters
 * - A delta value is added into VelocityTracker and a new speed is calculated.
 * - If the current speed is bigger than the previous one,  this value is remembered as
 * a latest fling speed with a timestamp
 * - After each scroll event a fling countdown starts ( by default 70ms) which
 * resets if new scroll event is received
 * - If fling countdown is finished - it means that the finger was probably raised from RSB, there will be no other events and probably
 * this is the last event during this session. After it a fling is triggered.
 * - Fling is stopped when a new scroll event happens
 *
 * The screen containing the scrollable item should request the focus
 * by calling [requestFocus] method
 *
 * ```
 * LaunchedEffect(Unit) {
 *   focusRequester.requestFocus()
 * }
 * ```
 * @param focusRequester Requests the focus for rotary input
 * @param scrollableState Scrollable state which will be scrolled while receiving rotary events
 * @param flingBehavior Logic describing fling behavior.
 * @param rotaryHaptics Class which will handle haptic feedback
 * @param reverseDirection Reverse the direction of scrolling. Should be aligned with
 * Scrollable `reverseDirection` parameter
 */
@ExperimentalHorologistApi
@Suppress("ComposableModifierFactory")
@Composable
public fun Modifier.rotaryWithFling(
    focusRequester: FocusRequester,
    scrollableState: ScrollableState,
    flingBehavior: FlingBehavior = ScrollableDefaults.flingBehavior(),
    rotaryHaptics: RotaryHapticHandler = rememberRotaryHapticHandler(scrollableState),
    reverseDirection: Boolean = false
): Modifier = rotaryHandler(
    rotaryScrollHandler = RotaryDefaults.rememberFlingHandler(scrollableState, flingBehavior),
    reverseDirection = reverseDirection,
    rotaryHaptics = rotaryHaptics
)
    .focusRequester(focusRequester)
    .focusable()

/**
 * A modifier which connects rotary events with scrollable.
 * This modifier only supports scroll without fling or snap.
 * The screen containing the scrollable item should request the focus
 * by calling [requestFocus] method
 *
 * ```
 * LaunchedEffect(Unit) {
 *   focusRequester.requestFocus()
 * }
 * ```
 * @param focusRequester Requests the focus for rotary input
 * @param scrollableState Scrollable state which will be scrolled while receiving rotary events
 * @param rotaryHaptics Class which will handle haptic feedback
 * @param reverseDirection Reverse the direction of scrolling. Should be aligned with
 * Scrollable `reverseDirection` parameter
 */
@ExperimentalHorologistApi
@Suppress("ComposableModifierFactory")
@Composable
public fun Modifier.rotaryWithScroll(
    focusRequester: FocusRequester,
    scrollableState: ScrollableState,
    rotaryHaptics: RotaryHapticHandler = rememberRotaryHapticHandler(scrollableState),
    reverseDirection: Boolean = false
): Modifier = rotaryHandler(
    rotaryScrollHandler = RotaryDefaults.rememberFlingHandler(scrollableState, null),
    reverseDirection = reverseDirection,
    rotaryHaptics = rotaryHaptics
)
    .focusRequester(focusRequester)
    .focusable()

/**
 * A modifier which connects rotary events with scrollable.
 * This modifier supports snap.
 *
 * The screen containing the scrollable item should request the focus
 * by calling [requestFocus] method
 *
 * ```
 * LaunchedEffect(Unit) {
 *   focusRequester.requestFocus()
 * }
 * ```
 * @param focusRequester Requests the focus for rotary input
 * @param rotaryScrollAdapter A connection between scrollable objects and rotary events
 * @param rotaryHaptics Class which will handle haptic feedback
 * @param reverseDirection Reverse the direction of scrolling. Should be aligned with
 * Scrollable `reverseDirection` parameter
 */
@ExperimentalHorologistApi
@Suppress("ComposableModifierFactory")
@Composable
public fun Modifier.rotaryWithSnap(
    focusRequester: FocusRequester,
    rotaryScrollAdapter: RotaryScrollAdapter,
    rotaryHaptics: RotaryHapticHandler = rememberRotaryHapticHandler(rotaryScrollAdapter.scrollableState),
    reverseDirection: Boolean = false
): Modifier = rotaryHandler(
    rotaryScrollHandler = RotaryDefaults.rememberSnapHandler(rotaryScrollAdapter),
    reverseDirection = reverseDirection,
    rotaryHaptics = rotaryHaptics
)
    .focusRequester(focusRequester)
    .focusable()

/**
 * An extension function for creating [RotaryScrollAdapter] from [ScalingLazyListState]
 */
@ExperimentalHorologistApi
public fun ScalingLazyListState.toRotaryScrollAdapter(): RotaryScrollAdapter =
    ScalingLazyColumnRotaryScrollAdapter(this)

/**
 * An implementation of rotary scroll adapter for [ScalingLazyColumn]
 */
@ExperimentalHorologistApi
public class ScalingLazyColumnRotaryScrollAdapter(
    override val scrollableState: ScalingLazyListState
) : RotaryScrollAdapter {

    /**
     * Calculates an average height of an item by taking an average from visible items height.
     */
    override fun averageItemSize(): Float {
        val visibleItems = scrollableState.layoutInfo.visibleItemsInfo
        return (visibleItems.fastSumBy { it.unadjustedSize } / visibleItems.size).toFloat()
    }

    /**
     * Returns a size of currently selected item
     */
    override fun currentItemSize(): Float =
        scrollableState.layoutInfo.visibleItemsInfo
            .first { it.index == currentItemIndex() }.size.toFloat()

    /**
     * Current (centred) item index
     */
    override fun currentItemIndex(): Int = scrollableState.centerItemIndex

    /**
     * An offset from the item centre
     */
    override fun currentItemOffset(): Float = scrollableState.centerItemScrollOffset.toFloat()
}

/**
 * An adapter which connects scrollableState to Rotary
 */
@ExperimentalHorologistApi
public interface RotaryScrollAdapter {

    /**
     * A scrollable state. Used for performing scroll when Rotary events received
     */
    @ExperimentalHorologistApi
    public val scrollableState: ScrollableState

    /**
     * Average size of an item. Used for estimating the scrollable distance
     */
    @ExperimentalHorologistApi
    public fun averageItemSize(): Float

    /**
     * Size of a current item.
     */
    @ExperimentalHorologistApi
    public fun currentItemSize(): Float

    /**
     * A current item index. Used for scrolling
     */
    @ExperimentalHorologistApi
    public fun currentItemIndex(): Int

    /**
     * An offset from the centre or the border of the current item.
     */
    @ExperimentalHorologistApi
    public fun currentItemOffset(): Float
}

/**
 * Defaults for rotary modifiers
 */
@ExperimentalHorologistApi
public object RotaryDefaults {

    /**
     * Handles scroll with fling.
     * @param scrollableState Scrollable state which will be scrolled while receiving rotary events
     * @param flingBehavior Logic describing Fling behavior. If null - fling will not happen
     * @param isLowRes Whether the input is Low-res (a bezel) or high-res(a crown/rsb)
     */
    @ExperimentalHorologistApi
    @Composable
    public fun rememberFlingHandler(
        scrollableState: ScrollableState,
        flingBehavior: FlingBehavior? = null,
    ): RotaryScrollHandler {
        val viewConfiguration = ViewConfiguration.get(LocalContext.current)

        return remember(scrollableState, flingBehavior) {
            fun rotaryFlingBehavior() = flingBehavior?.run {
                RotaryFlingBehavior(
                    scrollableState, flingBehavior, viewConfiguration,
                    flingTimeframe = highResFlingTimeframe
                )
            }

            fun scrollBehavior() = AnimationScrollBehavior(scrollableState)

            RotaryScrollFlingHandler(
                rotaryFlingBehaviorFactory = { rotaryFlingBehavior() },
                scrollBehaviorFactory = { scrollBehavior() }
            )
        }
    }

    /**
     * Handles scroll with snap
     * @param rotaryScrollAdapter A connection between scrollable objects and rotary events
     * @param snapParameters Snap parameters
     * @param isLowRes Whether the input is Low-res (a bezel) or high-res(a crown/rsb)
     */
    @ExperimentalHorologistApi
    @Composable
    public fun rememberSnapHandler(
        rotaryScrollAdapter: RotaryScrollAdapter,
        snapParameters: SnapParameters = snapParametersDefault(),
        isLowRes: Boolean = isLowResInput()
    ): RotaryScrollHandler = remember(rotaryScrollAdapter, snapParameters) {
        RotaryScrollSnapHandler(
            snapBehaviourFactory = {
                RotarySnapBehavior(rotaryScrollAdapter, snapParameters)
            },
            scrollBehaviourFactory = {
                AnimationScrollBehavior(rotaryScrollAdapter.scrollableState)
            },
            isLowRes = isLowRes
        )
    }

    /**
     * Returns default [SnapParameters]
     */
    @ExperimentalHorologistApi
    public fun snapParametersDefault(): SnapParameters =
        SnapParameters(
            snapOffset = 0,
            innerScrollThreshold = 100f,
            minSizeForInnerScroll = 300f
        )

    /**
     * Returns whether the input is Low-res (a bezel) or high-res(a crown/rsb).
     */
    @ExperimentalHorologistApi
    @Composable
    public fun isLowResInput(): Boolean = LocalContext.current.packageManager
        .hasSystemFeature("android.hardware.rotaryencoder.lowres")

    private const val highResFlingTimeframe: Long = 30L
}

/**
 * Parameters used for snapping
 *
 * @param snapOffset an optional offset to be applied when snapping the item. After the snap the
 * snapped items offset will be [snapOffset].
 * @param innerScrollThreshold a threshold for an inner scroll after which snap happens.
 * Considered when the scroll reaches the edge of the item if item is
 * higher than [minSizeForInnerScroll]
 * @param minSizeForInnerScroll a minimum size of an item after which an inner scroll
 * will be happening.
 */
public class SnapParameters(
    public val snapOffset: Int,
    public val innerScrollThreshold: Float = 100f,
    public val minSizeForInnerScroll: Float = 300f
) {
    /**
     * Returns a snapping offset in [Dp]
     */
    @Composable
    public fun snapOffsetDp(): Dp {
        return with(LocalDensity.current) {
            snapOffset.toDp()
        }
    }
}

/**
 * An interface for handling scroll events
 */
@ExperimentalHorologistApi
public interface RotaryScrollHandler {
    /**
     * Handles scrolling events
     * @param coroutineScope A scope for performing async actions
     * @param event A scrollable event from rotary input, containing scrollable delta and timestamp
     * @param rotaryHaptics A haptics handler
     */
    @ExperimentalHorologistApi
    public suspend fun handleScrollEvent(
        coroutineScope: CoroutineScope, event: TimestampedDelta, rotaryHaptics: RotaryHapticHandler
    )
}

/**
 * An interface for scrolling behavior
 */
@ExperimentalHorologistApi
public interface RotaryScrollBehavior {
    /**
     * Handles scroll event to [targetValue]
     */
    @ExperimentalHorologistApi
    public suspend fun handleEvent(targetValue: Float)
}

/**
 * A helper class for flinging with rotary
 */
@ExperimentalHorologistApi
internal class RotaryFlingBehavior(
    private val scrollableState: ScrollableState,
    private val flingBehavior: FlingBehavior,
    viewConfiguration: ViewConfiguration,
    private val flingTimeframe: Long
) {

    // A time range during which the fling is valid.
    // For simplicity it's twice as long as [flingTimeframe]
    private val timeRangeToFling = flingTimeframe * 2

    //  A default fling factor for making fling slower
    private val flingScaleFactor = 0.7f

    private var previousVelocity = 0f

    private val rotaryVelocityTracker = RotaryVelocityTracker()

    private val minFlingSpeed = viewConfiguration.scaledMinimumFlingVelocity.toFloat()
    private val maxFlingSpeed = viewConfiguration.scaledMaximumFlingVelocity.toFloat()
    private var latestEventTimestamp: Long = 0

    private var flingVelocity: Float = 0f
    private var flingTimestamp: Long = 0

    /**
     * Starts a new fling tracking session
     * with specified timestamp
     */
    @ExperimentalHorologistApi
    fun startFlingTracking(timestamp: Long) {
        rotaryVelocityTracker.start(timestamp)
        latestEventTimestamp = timestamp
        previousVelocity = 0f
    }

    /**
     * Observing new event within a fling tracking session with new timestamp and delta
     */
    @ExperimentalHorologistApi
    fun observeEvent(timestamp: Long, delta: Float) {
        rotaryVelocityTracker.move(timestamp, delta)
        latestEventTimestamp = timestamp
    }

    /**
     * Performing fling if necessary and calling [beforeFling] before it is triggered
     */
    @ExperimentalHorologistApi
    suspend fun trackFling(beforeFling: () -> Unit) {
        val currentVelocity = rotaryVelocityTracker.velocity
        debugLog { "currentVelocity: $currentVelocity" }

        if (abs(currentVelocity) >= abs(previousVelocity)) {
            flingTimestamp = latestEventTimestamp
            flingVelocity = currentVelocity * flingScaleFactor
        }
        previousVelocity = currentVelocity

        // Waiting for a fixed amount of time before checking the fling
        delay(flingTimeframe)

        // For making a fling 2 criteria should be met:
        // 1) no more than
        // `rangeToFling` ms should pass between last fling detection
        // and the time of last motion event
        // 2) flingVelocity should exceed the minFlingSpeed
        debugLog {
            "Check fling:  flingVelocity: $flingVelocity " +
                "minFlingSpeed: $minFlingSpeed, maxFlingSpeed: $maxFlingSpeed"
        }
        if (latestEventTimestamp - flingTimestamp < timeRangeToFling &&
            abs(flingVelocity) > minFlingSpeed) {
            // Call beforeFling callback so we'll stop all other scrolls
            beforeFling()
            val velocity = flingVelocity.coerceIn(-maxFlingSpeed, maxFlingSpeed)
            scrollableState.scroll(MutatePriority.UserInput) {
                with(flingBehavior) {
                    debugLog { "Flinging with velocity $velocity" }
                    performFling(velocity)
                }
            }
        }
    }
}

/**
 * A rotary event object which contains a [timestamp] of the rotary event and a scrolled [delta].
 */
@ExperimentalHorologistApi
public data class TimestampedDelta(val timestamp: Long, val delta: Float)

/** Animation implementation of [RotaryScrollBehavior].
 * This class does a smooth animation when the scroll by N pixels is done.
 * This animation works well on Rsb(high-res) and Bezel(low-res) devices.
 */
@ExperimentalHorologistApi
public class AnimationScrollBehavior(
    private val scrollableState: ScrollableState
) : RotaryScrollBehavior {
    private var sequentialAnimation = false
    private var scrollAnimation = AnimationState(0f)
    private var prevPosition = 0f

    @ExperimentalHorologistApi
    override suspend fun handleEvent(targetValue: Float) {
        scrollableState.scroll(MutatePriority.UserInput) {
            debugLog { "ScrollAnimation value before start: ${scrollAnimation.value}" }

            scrollAnimation.animateTo(
                targetValue, animationSpec = spring(), sequentialAnimation = sequentialAnimation
            ) {
                val delta = value - prevPosition
                debugLog { "Animated by $delta, value: $value" }
                scrollBy(delta)
                prevPosition = value
                sequentialAnimation = value != this.targetValue
            }
        }
    }
}

/**
 * An animated implementation of [RotarySnapBehavior]. Uses animateScrollToItem
 * method for snapping to the Nth item
 */
@ExperimentalHorologistApi
public class RotarySnapBehavior(
    private val rotaryScrollAdapter: RotaryScrollAdapter,
    private val snapParameters: SnapParameters,
) {
    private var snapTarget: Int = 0
    private var sequentialSnap: Boolean = false

    private var anim = AnimationState(0f)
    private var expectedDistance = 0f

    private val defaultStiffness = 200f
    private var snapTargetUpdated = true

    /**
     * Preparing snapping. This method should be called before [startSnappingSession] is called.
     *
     * Snapping is done for current + [moveForElements] items.
     *
     * If [sequentialSnap] is true, items are summed up together.
     * For example, if [prepareSnapForItems] is called with
     * [moveForElements] = 2, 3, 5 -> then the snapping will happen to current + 10 items
     *
     * If [sequentialSnap] is false, then [moveForElements] are not summed up together.
     */
    @ExperimentalHorologistApi
    fun prepareSnapForItems(moveForElements: Int, sequentialSnap: Boolean) {
        this.sequentialSnap = sequentialSnap
        if (sequentialSnap) {
            snapTarget += moveForElements
        } else {
            snapTarget = (rotaryScrollAdapter.currentItemIndex() + moveForElements)
        }
        snapTarget = snapTarget.coerceAtLeast(0)
        debugLog { " Snap target updated" }
        snapTargetUpdated = true
    }

    fun isInnerScrollRange(event: TimestampedDelta): Boolean =
        rotaryScrollAdapter.run {
            val scrollRange = currentItemSize() / 2 - snapParameters.innerScrollThreshold
            currentItemSize() > snapParameters.minSizeForInnerScroll &&
                (currentItemOffset() + event.delta).absoluteValue < scrollRange
        }

    /**
     * A threshold after which snapping happens.
     */
    @ExperimentalHorologistApi
    fun snapThreshold(): Float = min(
        snapParameters.innerScrollThreshold * 2,
        rotaryScrollAdapter.currentItemSize()
    )

    suspend fun snapToClosestItem() {
        // Snapping to the closest item by using performFling method with 0 speed
        rotaryScrollAdapter.scrollableState.scroll(MutatePriority.UserInput) {
            debugLog { "snap to closest item" }
            var prevPosition = 0f
            AnimationState(0f).animateTo(
                targetValue = expectedDistanceToClosestItem(),
                animationSpec = tween(durationMillis = 200, easing = LinearOutSlowInEasing)
            ) {
                val animDelta = value - prevPosition
                scrollBy(animDelta)
                prevPosition = value
            }
            snapTarget = rotaryScrollAdapter.currentItemIndex()
        }
    }

    /**
     * Performs snapping to the specified in [prepareSnapForItems] element
     */
    public suspend fun snapToTargetItem() {
        if (sequentialSnap) {
            anim = anim.copy(0f)
        } else {
            anim = AnimationState(0f)
        }
        rotaryScrollAdapter.scrollableState.scroll(MutatePriority.UserInput) {
            // If snapTargetUpdated is true - then the target was updated so we
            // need to do snap again
            while (snapTargetUpdated) {
                snapTargetUpdated = false
                var latestCenterItem: Int
                var continueFirstScroll = true
                debugLog { "snapTarget $snapTarget" }
                while (continueFirstScroll) {
                    latestCenterItem = rotaryScrollAdapter.currentItemIndex()
                    anim = anim.copy(0f)
                    expectedDistance = expectedDistanceTo(snapTarget, snapParameters.snapOffset)
                    debugLog {
                        "expectedDistance = $expectedDistance, " +
                            "scrollableState.centerItemScrollOffset " +
                            "${rotaryScrollAdapter.currentItemOffset()}"
                    }
                    continueFirstScroll = false
                    var prevPosition = 0f

                    anim.animateTo(
                        targetValue = expectedDistance,
                        animationSpec = SpringSpec(
                            stiffness = defaultStiffness,
                            visibilityThreshold = 0.1f
                        ),
                        sequentialAnimation = (anim.velocity != 0f)
                    ) {
                        val animDelta = value - prevPosition
                        debugLog {
                            "First animation, value:$value, velocity:$velocity, animDelta:$animDelta"
                        }

                        // Exit animation if snap target was updated
                        if (snapTargetUpdated) cancelAnimation()

                        scrollBy(animDelta)
                        prevPosition = value

                        if (latestCenterItem != rotaryScrollAdapter.currentItemIndex()) {
                            continueFirstScroll = true
                            cancelAnimation()
                            return@animateTo
                        }

                        debugLog { "centerItemIndex =  ${rotaryScrollAdapter.currentItemIndex()}" }
                        if (rotaryScrollAdapter.currentItemIndex() == snapTarget) {
                            debugLog { "Target is visible. Cancelling first animation" }
                            debugLog {
                                "scrollableState.centerItemScrollOffset " + "${rotaryScrollAdapter.currentItemOffset()}"
                            }
//                            expectedDistance = -rotaryScrollAdapter.currentItemOffset()
                            expectedDistance = expectedDistanceToClosestItem()
                            continueFirstScroll = false
                            cancelAnimation()
                            return@animateTo
                        }
                    }
                }
                // Exit animation if snap target was updated
                if (snapTargetUpdated) continue

                anim = anim.copy(0f)
                var prevPosition = 0f
                anim.animateTo(
                    expectedDistance, animationSpec = SpringSpec(
                    stiffness = defaultStiffness, visibilityThreshold = 0.1f
                ), sequentialAnimation = (anim.velocity != 0f)
                ) {
                    // Exit animation if snap target was updated
                    if (snapTargetUpdated) cancelAnimation()

                    val animDelta = value - prevPosition
                    debugLog { "Final animation. velocity:$velocity, animDelta:$animDelta" }
                    scrollBy(animDelta)
                    prevPosition = value
                }
            }
        }
    }

    private fun expectedDistanceToClosestItem(): Float =
        rotaryScrollAdapter.run {
            if (currentItemSize() > snapParameters.minSizeForInnerScroll) {
                -sign(currentItemOffset()) *
                    (currentItemOffset().absoluteValue - currentItemSize() / 2 + snapParameters.innerScrollThreshold)
            } else {
                -currentItemOffset()
            }
        }

    private fun expectedDistanceTo(index: Int, targetScrollOffset: Int): Float {
        val averageSize = rotaryScrollAdapter.averageItemSize()
        val indexesDiff = index - rotaryScrollAdapter.currentItemIndex()
        debugLog { "Average size $averageSize" }
        return (averageSize * indexesDiff) + targetScrollOffset - rotaryScrollAdapter.currentItemOffset()
    }
}

/**
 * A modifier which handles rotary events.
 * It accepts ScrollHandler as the input - a class where main logic about how
 * scroll should be handled is lying
 */
@ExperimentalHorologistApi
@OptIn(ExperimentalComposeUiApi::class)
public fun Modifier.rotaryHandler(
    rotaryScrollHandler: RotaryScrollHandler,
    batchTimeframe: Long = 0L,
    reverseDirection: Boolean,
    rotaryHaptics: RotaryHapticHandler
): Modifier = composed {
    val channel = rememberTimestampChannel()
    val eventsFlow = remember(channel) { channel.receiveAsFlow() }

    composed {
        LaunchedEffect(eventsFlow) {
            eventsFlow
                // TODO: batching causes additional delays.
                // Do we really need to do this on this level?
                .batchRequestsWithinTimeframe(batchTimeframe).collectLatest {
                    debugLog {
                        "Scroll event received: " + "delta:${it.delta}, timestamp:${it.timestamp}"
                    }
                    rotaryScrollHandler.handleScrollEvent(this, it, rotaryHaptics)
                }
        }
        this.onRotaryScrollEvent {
            // Okay to ignore the ChannelResult returned from trySend because it is conflated
            // (see rememberTimestampChannel()).
            @Suppress("UNUSED_VARIABLE") val unused = channel.trySend(
                TimestampedDelta(
                    it.uptimeMillis, it.verticalScrollPixels * if (reverseDirection) -1f else 1f
                )
            )
            true
        }
    }
}

/**
 * Batching requests for scrolling events. This function combines all events together
 * (except first) within specified timeframe. Should help with performance on high-res devices.
 */
@ExperimentalHorologistApi
@OptIn(ExperimentalCoroutinesApi::class)
public fun Flow<TimestampedDelta>.batchRequestsWithinTimeframe(timeframe: Long): Flow<TimestampedDelta> {
    var delta = 0f
    var lastTimestamp = -timeframe
    return if (timeframe == 0L) {
        this
    } else {
        this.transformLatest {
            delta += it.delta
            debugLog { "Batching requests. delta:$delta" }
            if (lastTimestamp + timeframe <= it.timestamp) {
                lastTimestamp = it.timestamp
                debugLog { "No events before, delta= $delta" }
                emit(TimestampedDelta(it.timestamp, delta))
            } else {
                delay(timeframe)
                debugLog { "After delay, delta= $delta" }
                if (delta > 0f) {
                    emit(TimestampedDelta(it.timestamp, delta))
                }
            }
            delta = 0f
        }
    }
}

/**
 * A scroll handler for RSB(high-res) without snapping and with or without fling
 * A list is scrolled by the number of pixels received from the rotary device.
 *
 * This class is a little bit different from LowResScrollHandler class - it has a filtering
 * for events which are coming with wrong sign ( this happens to rsb devices,
 * especially at the end of the scroll)
 *
 * This scroll handler supports fling. It can be set with [RotaryFlingBehavior].
 */
internal class RotaryScrollFlingHandler(
    private val rotaryFlingBehaviorFactory: () -> RotaryFlingBehavior?,
    private val scrollBehaviorFactory: () -> RotaryScrollBehavior,
) : RotaryScrollHandler {

    private val gestureThresholdTime = 200L
    private var previousScrollEventTime = 0L
    private var rotaryScrollDistance = 0f

    private var scrollJob: Job = CompletableDeferred<Unit>()
    private var flingJob: Job = CompletableDeferred<Unit>()

    private var rotaryFlingBehavior: RotaryFlingBehavior? = rotaryFlingBehaviorFactory()
    private var scrollBehavior: RotaryScrollBehavior = scrollBehaviorFactory()

    override suspend fun handleScrollEvent(
        coroutineScope: CoroutineScope, event: TimestampedDelta, rotaryHaptics: RotaryHapticHandler
    ) {
        val time = event.timestamp
        val isOppositeScrollValue = isOppositeValueAfterScroll(event.delta)

        if (isNewScrollEvent(time)) {
            debugLog { "New scroll event" }
            resetTracking(time)
            rotaryScrollDistance = event.delta
        } else {
            // Due to the physics of Rotary side button, some events might come
            // with an opposite axis value - either at the start or at the end of the motion.
            // We don't want to use these values for fling calculations.
            if (!isOppositeScrollValue) {
                rotaryFlingBehavior?.observeEvent(event.timestamp, event.delta)
            } else {
                debugLog { "Opposite value after scroll :${event.delta}" }
            }
            rotaryScrollDistance += event.delta
        }

        scrollJob.cancel()

        rotaryHaptics.handleScrollHaptic(event.delta)
        debugLog { "Rotary scroll distance: $rotaryScrollDistance" }

        previousScrollEventTime = time
        scrollJob = coroutineScope.async {
            scrollBehavior.handleEvent(rotaryScrollDistance)
        }

        if (rotaryFlingBehavior != null) {
            flingJob.cancel()
            flingJob = coroutineScope.async {
                rotaryFlingBehavior?.trackFling(beforeFling = {
                    debugLog { "Calling before fling section" }
                    scrollJob.cancel()
                    scrollBehavior = scrollBehaviorFactory()
                })
            }
        }
    }

    private fun isOppositeValueAfterScroll(delta: Float): Boolean = sign(rotaryScrollDistance) * sign(
        delta
    ) == -1f && (abs(delta) < abs(rotaryScrollDistance))

    private fun isNewScrollEvent(timestamp: Long): Boolean {
        val timeDelta = timestamp - previousScrollEventTime
        return previousScrollEventTime == 0L || timeDelta > gestureThresholdTime
    }

    private fun resetTracking(timestamp: Long) {
        scrollBehavior = scrollBehaviorFactory()
        rotaryFlingBehavior = rotaryFlingBehaviorFactory()
        rotaryFlingBehavior?.startFlingTracking(timestamp)
    }
}

/**
 * A scroll handler with snapping and without fling
 * Snapping happens after a threshold is reached ( set in [RotarySnapBehavior])
 *
 * This scroll handler doesn't support fling.
 */
internal class RotaryScrollSnapHandler(
    val snapBehaviourFactory: () -> RotarySnapBehavior,
    val scrollBehaviourFactory: () -> RotaryScrollBehavior,
    val isLowRes: Boolean
) : RotaryScrollHandler {

    private val gestureThresholdTime = 200L
    private val snapDelay = 100L
    private val resistanceFactor = 4f
    private var scrollJob: Job = CompletableDeferred<Unit>()
    private var snapJob: Job = CompletableDeferred<Unit>()

    private var previousScrollEventTime = 0L
    private var snapAccumulator = 0f
    private var rotaryScrollDistance = 0f

    private var snapBehaviour = snapBehaviourFactory()
    private var scrollBehaviour = scrollBehaviourFactory()

    private var innerScrollHappened = false

    override suspend fun handleScrollEvent(
        coroutineScope: CoroutineScope, event: TimestampedDelta, rotaryHaptics: RotaryHapticHandler
    ) {
        val time = event.timestamp

        if (snapBehaviour.isInnerScrollRange(event)) {

            if (isNewScrollEvent(time)) {
                debugLog { "New non-snap scroll" }
                scrollBehaviour = scrollBehaviourFactory()
                snapAccumulator = 0f
                rotaryScrollDistance = 0f
            }
            if (!innerScrollHappened) {
                debugLog { "Resetting innerScrollHappened" }
                innerScrollHappened = true
                scrollBehaviour = scrollBehaviourFactory()
                rotaryScrollDistance = 0f
            }

            rotaryScrollDistance += event.delta
            previousScrollEventTime = time

            debugLog { "Non-snap scroll for $rotaryScrollDistance" }

            rotaryHaptics.handleScrollHaptic(event.delta)
            scrollJob.cancel()
            scrollJob = coroutineScope.async {
                scrollBehaviour.handleEvent(rotaryScrollDistance)
            }

        } else {
            if (isNewScrollEvent(time)) {
                debugLog { "New scroll event" }
                innerScrollHappened = false
                snapJob.cancel()
                snapBehaviour = snapBehaviourFactory()
                scrollBehaviour = scrollBehaviourFactory()
                snapAccumulator = event.delta
                rotaryScrollDistance = event.delta
            } else {
                if (innerScrollHappened) {
                    innerScrollHappened = false
                    scrollBehaviour = scrollBehaviourFactory()
                    rotaryScrollDistance = 0f
                }
                snapAccumulator += event.delta
                if (!snapJob.isActive) {
                    rotaryScrollDistance += event.delta
                }
            }
            debugLog { "Snap accumulator: $snapAccumulator" }
            debugLog { "Rotary scroll distance: $rotaryScrollDistance" }
            previousScrollEventTime = time

            if ((isLowRes && !snapBehaviour.isInnerScrollRange(event)) ||
                abs(snapAccumulator) > snapBehaviour.snapThreshold()) {

                debugLog { "Snap threshold reached" }
                scrollBehaviour = scrollBehaviourFactory()
                scrollJob.cancel()

                val snapDistance = sign(snapAccumulator).toInt()
                rotaryHaptics.handleSnapHaptic(event.delta)

                val sequentialSnap = snapJob.isActive
                debugLog {
                    "Prepare snap: snapDistance:$snapDistance, " + "sequentialSnap: $sequentialSnap"
                }
                snapBehaviour.prepareSnapForItems(snapDistance, sequentialSnap)
                if (!snapJob.isActive) {
                    snapJob.cancel()
                    snapJob = coroutineScope.async {
                        debugLog { "Snap started" }
                        try {
                            snapBehaviour.snapToTargetItem()
                        } finally {
                            debugLog { "Snap called finally" }
                        }
                    }
                }
                snapAccumulator = 0f
                rotaryScrollDistance = 0f
            } else {
                if (!snapJob.isActive) {
                    scrollJob.cancel()
                    // Making visible scroll slower by adding a resistance .
                    debugLog { "Scrolling for $rotaryScrollDistance/$resistanceFactor px" }
                    scrollJob = coroutineScope.async {
                        scrollBehaviour.handleEvent(rotaryScrollDistance / resistanceFactor)
                    }
                    delay(snapDelay)
                    debugLog { "Snap delay passed" }
                    scrollBehaviour = scrollBehaviourFactory()
                    rotaryScrollDistance = 0f
                    snapAccumulator = 0f
                    snapBehaviour.snapToClosestItem()
                }
            }
        }
    }

    private fun isNewScrollEvent(timestamp: Long): Boolean {
        val timeDelta = timestamp - previousScrollEventTime
        return previousScrollEventTime == 0L || timeDelta > gestureThresholdTime
    }

}

@Composable
private fun rememberTimestampChannel() = remember {
    Channel<TimestampedDelta>(capacity = Channel.CONFLATED)
}
