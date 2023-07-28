/*
 * Copyright 2023 The Android Open Source Project
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

package com.google.android.horologist.composables.picker

import com.google.android.horologist.compose.rotaryinput.RotaryScrollAdapter
import androidx.compose.ui.util.fastSumBy
import androidx.wear.compose.foundation.lazy.ScalingLazyListState
import com.google.android.horologist.annotations.ExperimentalHorologistApi
import com.google.android.horologist.compose.rotaryinput.ScalingLazyColumnRotaryScrollAdapter
import com.google.android.horologist.compose.rotaryinput.toRotaryScrollAdapter

internal fun PickerState.toRotaryScrollAdapter(): PickerRotaryScrollAdapter =
    PickerRotaryScrollAdapter(this)

internal class PickerRotaryScrollAdapter(
    override val scrollableState: PickerState
) : RotaryScrollAdapter {

    override fun averageItemSize(): Float {
        val visibleItems = scrollableState.scalingLazyListState
            .layoutInfo.visibleItemsInfo
        return (visibleItems.fastSumBy { it.unadjustedSize } / visibleItems.size).toFloat()
    }

    override fun currentItemIndex(): Int =
        scrollableState.scalingLazyListState.centerItemIndex

    override fun currentItemOffset(): Float =
        scrollableState.scalingLazyListState.centerItemScrollOffset.toFloat()
}
