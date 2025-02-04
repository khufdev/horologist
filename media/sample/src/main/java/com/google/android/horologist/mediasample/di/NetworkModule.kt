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

package com.google.android.horologist.mediasample.di

import android.content.Context
import android.net.ConnectivityManager
import android.net.wifi.WifiManager
import coil.ImageLoader
import coil.decode.SvgDecoder
import coil.disk.DiskCache
import coil.request.CachePolicy
import coil.util.DebugLogger
import com.google.android.horologist.mediasample.BuildConfig
import com.google.android.horologist.mediasample.data.api.UampService
import com.google.android.horologist.mediasample.data.api.WearArtworkUampService
import com.google.android.horologist.mediasample.ui.AppConfig
import com.google.android.horologist.networks.data.DataRequestRepository
import com.google.android.horologist.networks.data.InMemoryDataRequestRepository
import com.google.android.horologist.networks.data.RequestType
import com.google.android.horologist.networks.highbandwidth.HighBandwidthNetworkMediator
import com.google.android.horologist.networks.highbandwidth.StandardHighBandwidthNetworkMediator
import com.google.android.horologist.networks.logging.NetworkStatusLogger
import com.google.android.horologist.networks.okhttp.NetworkAwareCallFactory
import com.google.android.horologist.networks.okhttp.NetworkSelectingCallFactory
import com.google.android.horologist.networks.okhttp.impl.NetworkLoggingEventListenerFactory
import com.google.android.horologist.networks.request.NetworkRequester
import com.google.android.horologist.networks.request.NetworkRequesterImpl
import com.google.android.horologist.networks.rules.NetworkingRulesEngine
import com.google.android.horologist.networks.status.NetworkRepository
import com.google.android.horologist.networks.status.NetworkRepositoryImpl
import com.squareup.moshi.Moshi
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import okhttp3.Cache
import okhttp3.Call
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.logging.LoggingEventListener
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.io.File
import javax.inject.Provider
import javax.inject.Singleton
import kotlin.time.Duration.Companion.seconds

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Singleton
    @Provides
    fun networkRepository(
        @ApplicationContext application: Context,
        @ForApplicationScope coroutineScope: CoroutineScope
    ): NetworkRepository = NetworkRepositoryImpl.fromContext(
        application,
        coroutineScope
    )

    @Singleton
    @Provides
    fun cache(
        @ApplicationContext application: Context
    ): Cache = Cache(
        application.cacheDir.resolve("HttpCache"),
        10_000_000
    )

    @Singleton
    @Provides
    fun alwaysHttpsInterceptor(): Interceptor = Interceptor {
        var request = it.request()

        if (request.url.scheme == "http") {
            request = request.newBuilder().url(
                request.url.newBuilder().scheme("https").build()
            ).build()
        }

        it.proceed(request)
    }

    @Singleton
    @Provides
    fun okhttpClient(
        cache: Cache,
        alwaysHttpsInterceptor: Interceptor
    ): OkHttpClient {
        return OkHttpClient.Builder().followSslRedirects(false)
            .addInterceptor(alwaysHttpsInterceptor)
            .eventListenerFactory(LoggingEventListener.Factory()).cache(cache).build()
    }

    @Provides
    fun networkLogger(): NetworkStatusLogger = NetworkStatusLogger.Logging

    @Singleton
    @Provides
    fun dataRequestRepository(): DataRequestRepository =
        InMemoryDataRequestRepository()

    @Provides
    fun networkingRulesEngine(
        networkRepository: NetworkRepository,
        networkLogger: NetworkStatusLogger,
        appConfig: AppConfig
    ): NetworkingRulesEngine = NetworkingRulesEngine(
        networkRepository = networkRepository,
        logger = networkLogger,
        networkingRules = appConfig.strictNetworking!!
    )

    @Provides
    fun connectivityManager(
        @ApplicationContext application: Context
    ): ConnectivityManager =
        application.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    @Provides
    fun wifiManager(
        @ApplicationContext application: Context
    ): WifiManager = application.getSystemService(Context.WIFI_SERVICE) as WifiManager

    @Singleton
    @Provides
    fun aggregatingHighBandwidthRequester(
        networkLogger: NetworkStatusLogger,
        networkRequester: NetworkRequester,
        @ForApplicationScope coroutineScope: CoroutineScope
    ) = StandardHighBandwidthNetworkMediator(
        networkLogger,
        networkRequester,
        coroutineScope,
        3.seconds
    )

    @Singleton
    @Provides
    fun highBandwidthRequester(
        highBandwidthNetworkMediator: StandardHighBandwidthNetworkMediator
    ): HighBandwidthNetworkMediator = highBandwidthNetworkMediator

    @Singleton
    @Provides
    fun networkRequester(
        connectivityManager: ConnectivityManager
    ): NetworkRequester = NetworkRequesterImpl(
        connectivityManager
    )

    @Singleton
    @Provides
    fun networkAwareCallFactory(
        appConfig: AppConfig,
        okhttpClient: OkHttpClient,
        networkingRulesEngine: Provider<NetworkingRulesEngine>,
        highBandwidthNetworkMediator: Provider<HighBandwidthNetworkMediator>,
        dataRequestRepository: DataRequestRepository,
        networkRepository: NetworkRepository,
        @ForApplicationScope coroutineScope: CoroutineScope,
        logger: NetworkStatusLogger
    ): Call.Factory =
        if (appConfig.strictNetworking != null) {
            NetworkSelectingCallFactory(
                networkingRulesEngine.get(),
                highBandwidthNetworkMediator.get(),
                networkRepository,
                dataRequestRepository,
                okhttpClient,
                coroutineScope,
                logger = logger
            )
        } else {
            okhttpClient.newBuilder()
                .eventListenerFactory(
                    NetworkLoggingEventListenerFactory(
                        logger,
                        networkRepository,
                        okhttpClient.eventListenerFactory,
                        dataRequestRepository
                    )
                )
                .build()
        }

    @Singleton
    @Provides
    fun moshi() = Moshi.Builder().build()

    @Singleton
    @Provides
    fun mooshiConverterFactory(
        moshi: Moshi
    ): MoshiConverterFactory = MoshiConverterFactory.create(moshi)

    @Singleton
    @Provides
    fun retrofit(
        callFactory: Call.Factory,
        moshiConverterFactory: MoshiConverterFactory
    ) =
        Retrofit.Builder()
            .addConverterFactory(moshiConverterFactory)
            .baseUrl(UampService.BASE_URL)
            .callFactory(
                NetworkAwareCallFactory(
                    callFactory,
                    RequestType.ApiRequest
                )
            ).build()

    @Singleton
    @Provides
    fun uampService(
        retrofit: Retrofit
    ): UampService = WearArtworkUampService(
        retrofit.create(UampService::class.java)
    )

    @Singleton
    @Provides
    fun imageLoader(
        @ApplicationContext application: Context,
        @CacheDir cacheDir: File,
        callFactory: Call.Factory
    ): ImageLoader = ImageLoader.Builder(application)
        .crossfade(false)
        .components {
            add(SvgDecoder.Factory())
        }
        .respectCacheHeaders(false).diskCache {
            DiskCache.Builder()
                .directory(cacheDir.resolve("image_cache"))
                .build()
        }
        .memoryCachePolicy(CachePolicy.ENABLED)
        .diskCachePolicy(CachePolicy.ENABLED)
        .networkCachePolicy(CachePolicy.ENABLED)
        .callFactory {
            NetworkAwareCallFactory(
                callFactory,
                defaultRequestType = RequestType.ImageRequest
            )
        }.apply {
            if (BuildConfig.DEBUG) {
                logger(DebugLogger())
            }
        }.build()
}
